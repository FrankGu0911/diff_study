import torch,os,logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

class LCDiffPlannerTrainer:
    def __init__(self,
                gru_model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                lr_scheduler:torch.optim.lr_scheduler=None,
                with_rgb:bool=False,
                with_lidar:bool=False,
                with_stop_reason:bool=False,
                autocast:bool=False,
                writer:SummaryWriter=None,
                model_save_path:str='pretrained/controlnet',
                dist:bool=False,
                half:bool=False):
        if dist:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        self.autocast = autocast
        self.half = half
        if self.half and self.autocast:
            logging.warning("Half precision is enabled, autocast will be disabled.")
            self.autocast = False
        if self.half:
            self.gru = gru_model.to(torch.bfloat16)
        else:
            self.gru = gru_model
        self.gru = self.gru.cuda(self.gpu_id)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dist = dist
        if dist:
            self.gru = DDP(self.gru,device_ids=[self.gpu_id])
        self.autocast = autocast
        self.writer = writer
        self.model_save_path = model_save_path
        self.with_rgb = with_rgb
        self.with_lidar = with_lidar
        self.with_stop_reason = with_stop_reason
        self.criterion = torch.nn.MSELoss()

    def train_one_epoch(self,current_epoch:int):
        logging.info(f"[GPU:{self.gpu_id}] Epoch {current_epoch} | Train Steps: {len(self.train_loader)}")
        self.gru.train()
        if self.gpu_id == 0:
            train_loop = tqdm(self.train_loader,
                              desc="Train Epoch {}".format(current_epoch),
                              total=len(self.train_loader),
                              smoothing=0)
        else:
            train_loop = self.train_loader
        train_loss = []
        if self.with_stop_reason:
            train_loss_wp = []
            train_loss_sr = []
        scaler = GradScaler()
        for i,(data,label) in enumerate(train_loop):
            # topdown_feature: (bs,4,32,32)
            # measurement_feature: (bs,2+6)
            # label: (bs,4,4,2)
            topdown_feature, measurement_feature = data[0],data[1]
            wp = label[0]
            rgb_feature = None
            lidar_feature = None
            stop_reason = None
            if self.with_rgb:
                rgb_feature = data[2]
            if self.with_lidar:
                lidar_feature = data[3]
            if self.with_stop_reason:
                stop_reason = label[1]
            if self.half:
                topdown_feature = topdown_feature.to(torch.bfloat16).cuda(self.gpu_id)
                measurement_feature = measurement_feature.to(torch.bfloat16).cuda(self.gpu_id)
                wp = wp.to(torch.bfloat16).cuda(self.gpu_id)
                stop_reason = stop_reason.to(torch.bfloat16).cuda(self.gpu_id) if stop_reason is not None else None
                rgb_feature = rgb_feature.to(torch.bfloat16).cuda(self.gpu_id) if rgb_feature is not None else None
                lidar_feature = lidar_feature.to(torch.bfloat16).cuda(self.gpu_id) if lidar_feature is not None else None
            else:
                topdown_feature = topdown_feature.to(torch.float32).cuda(self.gpu_id)
                measurement_feature = measurement_feature.to(torch.float32).cuda(self.gpu_id)
                wp = wp.to(torch.float32).cuda(self.gpu_id)
                stop_reason = stop_reason.to(torch.float32).cuda(self.gpu_id) if stop_reason is not None else None
                rgb_feature = rgb_feature.to(torch.float32).cuda(self.gpu_id) if rgb_feature is not None else None
                lidar_feature = lidar_feature.to(torch.float32).cuda(self.gpu_id) if lidar_feature is not None else None
            with autocast(enabled=self.autocast):
                if self.with_rgb and self.with_lidar:
                    out = self.gru(topdown_feature,measurement_feature,
                                    rgb_feature=rgb_feature,
                                    lidar_feature=lidar_feature)
                elif self.with_rgb:
                    out = self.gru(topdown_feature,measurement_feature,
                                    rgb_feature=rgb_feature)
                elif self.with_lidar:
                    out = self.gru(topdown_feature,measurement_feature,
                                    lidar_feature=lidar_feature)
                else:
                    out = self.gru(topdown_feature,measurement_feature)
                if self.with_stop_reason:
                    out_wp, out_reason = out
                    loss_wp = self.criterion(out_wp,wp)
                    loss_sr = torch.nn.BCEWithLogitsLoss()(out_reason,stop_reason)
                    loss = loss_wp+ 10 * loss_sr
                else:
                    loss = self.criterion(out,wp)
            if torch.isnan(loss):
                tqdm.write("Loss is NaN!")
            else:
                train_loss.append(loss.item())
                if self.with_stop_reason:
                    train_loss_wp.append(loss_wp.item())
                    train_loss_sr.append(loss_sr.item())
                if self.autocast:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            self.optimizer.zero_grad()
            if self.gpu_id == 0:
                if len(train_loss) != 0:
                    avg_loss = sum(train_loss) / len(train_loss)
                    if self.with_stop_reason:
                        avg_loss_wp = sum(train_loss_wp) / len(train_loss_wp)
                        avg_loss_sr = sum(train_loss_sr) / len(train_loss_sr)
                        train_loop.set_postfix({"loss_wp":avg_loss_wp,"loss_sr":avg_loss_sr,"lr":"%.1e" %self.optimizer.param_groups[0]["lr"]})
                    else:
                        train_loop.set_postfix({"loss":avg_loss,"lr":"%.1e" %self.optimizer.param_groups[0]["lr"]})
            self.lr_scheduler.step(current_epoch + i / len(self.train_loader))
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("train_loss",sum(train_loss)/len(train_loss),current_epoch)
            if self.with_stop_reason:
                self.writer.add_scalar("train_loss_wp",sum(train_loss_wp)/len(train_loss_wp),current_epoch)
                self.writer.add_scalar("train_loss_sr",sum(train_loss_sr)/len(train_loss_sr),current_epoch)

    def val_one_epoch(self,current_epoch:int):
        self.gru.eval()
        if self.gpu_id == 0:
            val_loop = tqdm(self.val_loader,desc="Val Epoch {}".format(current_epoch),total=len(self.val_loader))
        else:
            val_loop = self.val_loader
        val_loss = []
        if self.with_stop_reason:
            val_loss_wp = []
            val_loss_sr = []
        for (data,label) in val_loop:
            topdown_feature, measurement_feature = data[0],data[1]
            wp = label[0]
            rgb_feature, lidar_feature = None, None
            stop_reason = None
            if self.with_rgb:
                rgb_feature = data[2]
            if self.with_lidar:
                lidar_feature = data[3]
            if self.with_stop_reason:
                stop_reason = label[1]
            if self.half:
                topdown_feature = topdown_feature.to(torch.bfloat16).cuda(self.gpu_id)
                measurement_feature = measurement_feature.to(torch.bfloat16).cuda(self.gpu_id)
                wp = wp.to(torch.bfloat16).cuda(self.gpu_id)
                stop_reason = stop_reason.to(torch.bfloat16).cuda(self.gpu_id) if stop_reason is not None else None
                rgb_feature = rgb_feature.to(torch.bfloat16).cuda(self.gpu_id) if rgb_feature is not None else None
                lidar_feature = lidar_feature.to(torch.bfloat16).cuda(self.gpu_id) if lidar_feature is not None else None
            else:
                topdown_feature = topdown_feature.to(torch.float32).cuda(self.gpu_id)
                measurement_feature = measurement_feature.to(torch.float32).cuda(self.gpu_id)
                wp = wp.to(torch.float32).cuda(self.gpu_id)
                stop_reason = stop_reason.to(torch.float32).cuda(self.gpu_id) if stop_reason is not None else None
                rgb_feature = rgb_feature.to(torch.float32).cuda(self.gpu_id) if rgb_feature is not None else None
                lidar_feature = lidar_feature.to(torch.float32).cuda(self.gpu_id) if lidar_feature is not None else None
            with torch.no_grad():
                if self.with_rgb and self.with_lidar:
                    out = self.gru(topdown_feature,measurement_feature,
                                    rgb_feature=rgb_feature,
                                    lidar_feature=lidar_feature)
                elif self.with_rgb:
                    out = self.gru(topdown_feature,measurement_feature,
                                    rgb_feature=rgb_feature)
                elif self.with_lidar:
                    out = self.gru(topdown_feature,measurement_feature,
                                    lidar_feature=lidar_feature)
                else:
                    out = self.gru(topdown_feature,measurement_feature)
                if self.with_stop_reason:
                    out, out_reason = out
                    loss_wp = self.criterion(out,wp)
                    loss_sr = torch.nn.BCEWithLogitsLoss()(out_reason,stop_reason)
                    loss = loss_wp+loss_sr
                else:
                    loss = self.criterion(out,wp)
            if torch.isnan(loss):
                tqdm.write("Loss is NaN!")
            else:
                val_loss.append(loss.item())
                if self.with_stop_reason:
                    val_loss_wp.append(loss_wp.item())
                    val_loss_sr.append(loss_sr.item())
            if self.gpu_id == 0:
                if len(val_loss) != 0:
                    avg_loss = sum(val_loss) / len(val_loss)
                    if self.with_stop_reason:
                        avg_loss_wp = sum(val_loss_wp) / len(val_loss_wp)
                        avg_loss_sr = sum(val_loss_sr) / len(val_loss_sr)
                        val_loop.set_postfix({"loss_wp":avg_loss_wp,"loss_sr":avg_loss_sr})
                    else:
                        val_loop.set_postfix({"loss":avg_loss})
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("val_loss",sum(val_loss)/len(val_loss),current_epoch)
            if self.with_stop_reason:
                self.writer.add_scalar("val_loss_wp",sum(val_loss_wp)/len(val_loss_wp),current_epoch)
                self.writer.add_scalar("val_loss_sr",sum(val_loss_sr)/len(val_loss_sr),current_epoch)
    
    def save_model(self,current_epoch:int,path:str):
        if self.dist:
            state = {
                "epoch":current_epoch,
                "model_state_dict":self.gru.module.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        else:
            state = {
                "epoch":current_epoch,
                "model_state_dict":self.gru.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        CheckPath(path)
        torch.save(state,os.path.join(path,"gru_model_%d.pth" %current_epoch))

    def train(self,current_epoch:int,max_epoch:int):
        if current_epoch >= max_epoch:
            logging.info(f"Current epoch {current_epoch} is greater than max epoch {max_epoch}, skip training.")
            return
        for epoch in range(current_epoch,max_epoch):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)
            if self.gpu_id == 0 and self.model_save_path is not None:
                self.save_model(epoch,self.model_save_path)