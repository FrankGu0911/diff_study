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
                autocast:bool=False,
                writer:SummaryWriter=None,
                model_save_path:str='pretrained/diffusion',
                dist:bool=True):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = unet_model.cuda(self.gpu_id)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        if self.gpu_id == 0:
            train_loop = tqdm(self.train_loader,desc="Train Epoch {}".format(current_epoch),total=len(self.train_loader))
        else:
            train_loop = self.train_loader
        train_loss = []
        scaler = GradScaler()
        for (data,label) in train_loop:
            # data -> (batch_size, 4, 768)
            # label -> (batch_size, 4, 32, 32)
            data = data.cuda(self.gpu_id)
            label = label.cuda(self.gpu_id)
            self.optimizer.zero_grad()
            noise = torch.randn_like(label)
            noise_step = torch.randint(0, 1000, (1, )).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label, noise, noise_step)
            if self.autocast:
                with autocast():
                    pred = self.model(z_noise,data,noise_step)
                    loss = self.criterion(pred,noise)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                pred = self.model(z_noise,data,noise_step)
                loss = self.criterion(pred,label)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            if self.gpu_id == 0:
                if len(train_loss) != 0:
                    avg_loss = sum(train_loss)/len(train_loss)
                train_loop.set_postfix({"loss":avg_loss})
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("train_loss",sum(train_loss)/len(train_loss),current_epoch)

    def val_one_epoch(self,current_epoch:int):
        self.model.eval()
        if self.gpu_id == 0:
            val_loop = tqdm(self.val_loader,desc="Val Epoch {}".format(current_epoch),total=len(self.val_loader))
        else:
            val_loop = self.val_loader
        val_loss = []
        for (data,label) in val_loop:
            data = data.cuda(self.gpu_id)
            label = label.cuda(self.gpu_id)
            noise = torch.randn_like(label)
            noise_step = torch.randint(0, 1000, (1, )).long().cuda(self.gpu_id)
            z_noise = self.scheduler.add_noise(label, noise, noise_step)
            with torch.no_grad():
                pred = self.model(z_noise,data,noise_step)
                loss = self.criterion(pred,noise)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    val_loss.append(loss.item())
                if self.gpu_id == 0:
                    if len(val_loss) != 0:
                        avg_loss = sum(val_loss)/len(val_loss)
                    val_loop.set_postfix({"loss":avg_loss})
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("val_loss",sum(val_loss)/len(val_loss),current_epoch)

    def save_checkpoint(self,epoch:int,path:str):
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