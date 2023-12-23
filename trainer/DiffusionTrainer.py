import torch,os,logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast
from diffusers import PNDMScheduler

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

class DiffusionTrainer:
    def __init__(self,
                unet_model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                lr_scheduler:torch.optim.lr_scheduler=None,
                with_lidar:bool=False,
                autocast:bool=False,
                accumulation:int=1,
                writer:SummaryWriter=None,
                model_save_path:str='pretrained/diffusion',
                interval_frame:int=1,
                dist:bool=True,
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
            self.model = unet_model.to(torch.bfloat16)
        else:
            self.model = unet_model
        self.model = self.model.cuda(self.gpu_id)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.with_lidar = with_lidar
        self.accumulation = accumulation
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dist = dist
        self.interval_frame = interval_frame
        if dist:
            self.model = DDP(self.model,device_ids=[self.gpu_id])
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
        self.model.train()
        # torch.manual_seed(2023)
        if self.gpu_id == 0:
            train_loop = tqdm(self.train_loader,
                              desc="Train Epoch {}".format(current_epoch),
                              total=len(self.train_loader),
                              smoothing=0)
        else:
            train_loop = self.train_loader
        train_loss = []
        cur_loss = []
        scaler = GradScaler()
        for i,(data,label) in enumerate(train_loop):
            # no_lidar
            # data -> (batch_size, 4, 768)
            # label -> (batch_size, 4, 32, 32)
            # with_lidar
            # data[0] -> (batch_size, 4, 768)
            # data[1] -> (batch_size, 3, 256, 256)
            # label -> (batch_size, 4, 32, 32)
            if i % self.interval_frame != 0:
                continue
            if self.with_lidar:
                data, lidar = data[0],data[1]
            else:
                data = data
            if self.half:
                data = data.to(torch.bfloat16).cuda(self.gpu_id)
                if self.with_lidar:
                    lidar = lidar.to(torch.bfloat16).cuda(self.gpu_id)
                label = label.to(torch.bfloat16).cuda(self.gpu_id)
            else:
                data = data.cuda(self.gpu_id)
                if self.with_lidar:
                    lidar = lidar.cuda(self.gpu_id)
                label = label.cuda(self.gpu_id)
            # self.optimizer.zero_grad()
            noise = torch.randn_like(label)
            noise_step = torch.randint(0, 1000, (data.shape[0], )).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label, noise, noise_step)
            if (current_epoch * len(self.train_loader) + i + 1) % self.accumulation != 0:
                if self.autocast:
                    with autocast(enabled=self.autocast) and self.model.no_sync():
                        if self.with_lidar:
                            pred = self.model(z_noise,data,noise_step,lidar)
                        else:
                            pred = self.model(z_noise,data,noise_step)
                        loss = self.criterion(pred,noise) 
                        scaler.scale(loss/self.accumulation).backward()
                else:
                    with self.model.no_sync():
                        if self.with_lidar:
                            pred = self.model(z_noise,data,noise_step,lidar)
                        else:
                            pred = self.model(z_noise,data,noise_step)
                        loss = self.criterion(pred,noise) 
                        (loss/self.accumulation).backward()
            else:
                if self.autocast:
                    with autocast(enabled=self.autocast):
                        if self.with_lidar:
                            pred = self.model(z_noise,data,noise_step,lidar)
                        else:
                            pred = self.model(z_noise,data,noise_step)
                        loss = self.criterion(pred,noise)
                        scaler.scale(loss/self.accumulation).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    if self.with_lidar:
                        pred = self.model(z_noise,data,noise_step,lidar)
                    else:
                        pred = self.model(z_noise,data,noise_step)
                    loss = self.criterion(pred,noise)
                    (loss/self.accumulation).backward()
                    self.optimizer.step()

                    # scaler.scale(loss/self.accumulation).backward()
                    # scaler.step(self.optimizer)
                    # scaler.update()
                self.optimizer.zero_grad()
            if torch.isnan(loss):
                tqdm.write("Loss is NaN!")
            else:
                train_loss.append(loss.item())
            if i % 100 == 0:
                cur_loss = []
            cur_loss.append(loss.item())
            if self.gpu_id == 0:
                if len(train_loss) != 0:
                    avg_loss = sum(train_loss)/len(train_loss)
                train_loop.set_postfix({"loss":avg_loss,"lr":"%.1e" %self.optimizer.param_groups[0]["lr"] })
            self.lr_scheduler.step(current_epoch + i / len(self.train_loader))
            if self.gpu_id == 0 and self.writer is not None and i % 100 == 0:
                self.writer.add_scalar("train_loss_%d" %current_epoch,sum(cur_loss)/len(cur_loss),i)
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("train_loss",sum(train_loss)/len(train_loss),current_epoch)

    def val_one_epoch(self,current_epoch:int):
        self.model.eval()
        # torch.manual_seed(2023)
        if self.gpu_id == 0:
            val_loop = tqdm(self.val_loader,desc="Val Epoch {}".format(current_epoch),total=len(self.val_loader))
        else:
            val_loop = self.val_loader
        val_loss = []
        cur_loss = []
        for i,(data,label) in enumerate(val_loop):
            if i % self.interval_frame != 0:
                continue
            if self.with_lidar:
                data, lidar = data[0].cuda(self.gpu_id),data[1]
            else:
                data = data
            if self.half:
                data = data.to(torch.bfloat16).cuda(self.gpu_id)
                if self.with_lidar:
                    lidar = lidar.to(torch.bfloat16).cuda(self.gpu_id)
                label = label.to(torch.bfloat16).cuda(self.gpu_id)
            else:
                data = data.cuda(self.gpu_id)
                if self.with_lidar:
                    lidar = lidar.cuda(self.gpu_id)
                label = label.cuda(self.gpu_id)
            noise = torch.randn_like(label)
            noise_step = torch.randint(0, 1000, (data.shape[0], )).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label, noise, noise_step)
            with torch.no_grad():
                if self.with_lidar:
                    pred = self.model(z_noise,data,noise_step,lidar)
                else:
                    pred = self.model(z_noise,data,noise_step)
                loss = self.criterion(pred,noise)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    val_loss.append(loss.item())
                if len(val_loss) % 100 == 0:
                    cur_loss = []
                cur_loss.append(loss.item())
                if self.gpu_id == 0:
                    if len(val_loss) != 0:
                        avg_loss = sum(val_loss)/len(val_loss)
                    val_loop.set_postfix({"loss":avg_loss})
            if self.gpu_id == 0 and self.writer is not None and len(val_loss) % 100 == 0:
                self.writer.add_scalar("val_loss_%d" %current_epoch,sum(cur_loss)/len(cur_loss),len(val_loss))
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("val_loss",sum(val_loss)/len(val_loss),current_epoch)

    def save_checkpoint(self,epoch:int,path:str):
        if self.dist:
            state = {
                "epoch":epoch,
                "model_state_dict":self.model.module.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        else:
            state = {
                "epoch":epoch,
                "model_state_dict":self.model.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict()
            }
        CheckPath(path)
        torch.save(state,os.path.join(path,f"diffusion_model_{epoch}.pth"))

    def train(self,current_epoch:int,max_epoch:int):
        if current_epoch >= max_epoch:
            logging.info(f"Current epoch {current_epoch} is greater than max epoch {max_epoch}, skip training.")
            return
        for epoch in range(current_epoch,max_epoch):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)
            if self.gpu_id == 0 and self.model_save_path is not None:
                self.save_checkpoint(epoch,self.model_save_path)

if __name__ == '__main__':
    pass