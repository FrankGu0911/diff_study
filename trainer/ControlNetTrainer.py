import torch,os,logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast
from diffusers import PNDMScheduler

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

class ControlNetTrainer:
    def __init__(self,
                unet_model:torch.nn.Module,
                controlnet_model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                lr_scheduler:torch.optim.lr_scheduler=None,
                autocast:bool=False,
                writer:SummaryWriter=None,
                model_save_path:str='pretrained/controlnet',
                dist:bool=False,
                half:bool=False,
                resume:bool=False):
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
            self.unet = unet_model.to(torch.bfloat16)
            self.controlnet = controlnet_model.to(torch.bfloat16)
        else:
            self.unet = unet_model
            self.controlnet = controlnet_model
        if not resume:
            logging.info("Start epoch 0, Load pretrained unet model.")
            self.controlnet.load_unet_param(self.unet)
        self.unet.requires_grad_(False)
        self.unet = self.unet.cuda(self.gpu_id)
        self.controlnet = self.controlnet.cuda(self.gpu_id)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dist = dist
        if dist:
            self.unet = DDP(self.unet,device_ids=[self.gpu_id])
            self.controlnet = DDP(self.controlnet,device_ids=[self.gpu_id])
        self.autocast = autocast
        self.writer = writer
        self.model_save_path = model_save_path
        self.scheduler = PNDMScheduler(
            num_train_timesteps=1000, 
            beta_end=0.012, 
            beta_start=0.00085,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            set_alpha_to_one=False,
            skip_prk_steps=True,
            steps_offset=1,
            trained_betas=None
            )
        self.criterion = torch.nn.MSELoss()

    def train_one_epoch(self,current_epoch:int):
        logging.info(f"[GPU:{self.gpu_id}] Epoch {current_epoch} | Train Steps: {len(self.train_loader)}")
        torch.manual_seed(2023)
        self.controlnet.train()
        self.unet.eval()
        if self.gpu_id == 0:
            train_loop = tqdm(self.train_loader,
                              desc="Train Epoch {}".format(current_epoch),
                              total=len(self.train_loader),
                              smoothing=0)
        else:
            train_loop = self.train_loader
        train_loss = []
        scaler = GradScaler()
        for i,(data,label) in enumerate(train_loop):
            # data: [(bs,4,768),(bs,3,256,256)]
            # label: (bs,4,32,32)
            data, lidar = data[0],data[1]
            if self.half:
                data = data.to(torch.bfloat16).cuda(self.gpu_id)
                lidar = lidar.to(torch.bfloat16).cuda(self.gpu_id)
                label = label.to(torch.bfloat16).cuda(self.gpu_id)
            else:
                data = data.cuda(self.gpu_id)
                lidar = lidar.cuda(self.gpu_id)
                label = label.cuda(self.gpu_id)
            noise = torch.randn_like(label)
            noise_step = torch.randint(0,1000,(data.shape[0],)).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label,noise,noise_step)
            if self.autocast:
                with autocast():
                    out_control_down, out_control_mid = self.controlnet(z_noise,data,noise_step,lidar)
                    out_unet = self.unet(z_noise,data,noise_step,
                                         down_block_additional_residuals = out_control_down,
                                         mid_block_additional_residual = out_control_mid)
                    loss = self.criterion(out_unet,noise)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
            else:
                out_control_down, out_control_mid = self.controlnet(z_noise,data,noise_step,lidar)
                out_unet = self.unet(z_noise,data,noise_step,
                                     down_block_additional_residuals = out_control_down,
                                     mid_block_additional_residual = out_control_mid)
                loss = self.criterion(out_unet,noise)
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
            if torch.isnan(loss):
                tqdm.write("Loss is NaN!")
            else:
                train_loss.append(loss.item())
            if self.gpu_id == 0:
                if len(train_loss) != 0:
                    avg_loss = sum(train_loss) / len(train_loss)
                train_loop.set_postfix({"loss":avg_loss,"lr":"%.1e" %self.optimizer.param_groups[0]["lr"] })
            self.lr_scheduler.step(current_epoch + i / len(self.train_loader))
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("train_loss",sum(train_loss)/len(train_loss),current_epoch)

    def val_one_epoch(self,current_epoch:int):
        self.controlnet.eval()
        self.unet.eval()
        torch.manual_seed(2023)
        if self.gpu_id == 0:
            val_loop = tqdm(self.val_loader,desc="Val Epoch {}".format(current_epoch),total=len(self.val_loader))
        else:
            val_loop = self.val_loader
        val_loss = []
        for (data,label) in val_loop:
            data, lidar = data[0],data[1]
            if self.half:
                data = data.to(torch.bfloat16).cuda(self.gpu_id)
                lidar = lidar.to(torch.bfloat16).cuda(self.gpu_id)
                label = label.to(torch.bfloat16).cuda(self.gpu_id)
            else:
                data = data.cuda(self.gpu_id)
                lidar = lidar.cuda(self.gpu_id)
                label = label.cuda(self.gpu_id)
            noise = torch.randn_like(label)
            noise_step = torch.randint(0,1000,(data.shape[0],)).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label,noise,noise_step)
            with torch.no_grad():
                out_control_down, out_control_mid = self.controlnet(z_noise,data,noise_step,lidar)
                out_unet = self.unet(z_noise,data,noise_step,
                                     down_block_additional_residuals = out_control_down,
                                     mid_block_additional_residual = out_control_mid)
                loss = self.criterion(out_unet,noise)
            if torch.isnan(loss):
                tqdm.write("Loss is NaN!")
            else:
                val_loss.append(loss.item())
            if self.gpu_id == 0:
                if len(val_loss) != 0:
                    avg_loss = sum(val_loss) / len(val_loss)
                val_loop.set_postfix({"loss":avg_loss})
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("val_loss",sum(val_loss)/len(val_loss),current_epoch)
    
    def save_model(self,current_epoch:int,path:str):
        if self.dist:
            state = {
                "epoch":current_epoch,
                "model_state_dict":self.controlnet.module.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        else:
            state = {
                "epoch":current_epoch,
                "model_state_dict":self.controlnet.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        CheckPath(path)
        torch.save(state,os.path.join(path,"controlnet_%d.pth" %current_epoch))

    def train(self,current_epoch:int,max_epoch:int):
        if current_epoch >= max_epoch:
            logging.info(f"Current epoch {current_epoch} is greater than max epoch {max_epoch}, skip training.")
            return
        for epoch in range(current_epoch,max_epoch):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)
            if self.gpu_id == 0 and self.model_save_path is not None:
                self.save_model(epoch,self.model_save_path)