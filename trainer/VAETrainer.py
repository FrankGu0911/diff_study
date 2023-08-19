import torch,os,logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.cuda.amp import GradScaler,autocast

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

def vae_loss(ori_x, con_x, mu, logvar):
    # the loss from the reconstruct image -> actually cannot describe the quality of the whole image
    bce_loss = torch.nn.functional.mse_loss(con_x.view(-1), ori_x.view(-1), reduction='sum')
    # how close the two distributions are (x and normal distribution)
    kl_diverage = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_diverage


class VAETrainer:
    def __init__(self,
                 vae_model:torch.nn.Module,
                 train_loader:torch.utils.data.DataLoader,
                 val_loader:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 lr_scheduler:torch.optim.lr_scheduler=None,
                 autocast:bool=False,
                 writer:SummaryWriter=None,
                 model_save_path:str=None):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = vae_model.cuda(self.gpu_id)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = DDP(self.model,device_ids=[self.gpu_id])
        self.autocast = autocast
        self.writer = writer
        self.model_save_path = model_save_path

    def train_one_epoch(self,current_epoch:int):
        logging.info(f"[GPU:{self.gpu_id}] Epoch {current_epoch} | Train Steps: {len(self.train_loader)}")
        self.model.train()
        if self.gpu_id == 0:
            train_loop = tqdm(self.train_loader,desc="Train Epoch {}".format(current_epoch),total=len(self.train_loader))
        else:
            train_loop = self.train_loader
        train_loss = []
        scaler = GradScaler()
        for i,(x,_) in enumerate(train_loop):
            x = x.cuda(self.gpu_id)
            self.optimizer.zero_grad()
            if self.autocast:
                with autocast():
                    con_x,mu,logvar = self.model(x)
                    loss = vae_loss(x,con_x,mu,logvar)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                con_x,mu,logvar = self.model(x)
                loss = vae_loss(x,con_x,mu,logvar)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step(current_epoch+i/len(self.train_loader))
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
        for (x,_) in val_loop:
            x = x.cuda(self.gpu_id)
            with torch.no_grad():
                con_x,mu,logvar = self.model(x)
                loss = vae_loss(x,con_x,mu,logvar)
                if torch.isnan(loss):
                    tqdm.write("Loss is NaN!")
                else:
                    val_loss.append(loss.item())
                if self.gpu_id == 0:
                    val_loop.set_postfix({"loss":loss.item()})
        if self.gpu_id == 0 and self.writer is not None:
            self.writer.add_scalar("val_loss",sum(val_loss)/len(val_loss),current_epoch)
            
    def save_checkpoint(self,epoch:int,path:str):
        state = {
            "epoch":epoch,
            "model_state_dict":self.model.module.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict()
        }
        CheckPath(path)
        torch.save(state,os.path.join(path,f"vae_model_{epoch}.pth"))
    
    def train(self,current_epoch:int,max_epoch:int):
        if current_epoch >= max_epoch:
            logging.info("Current epoch is greater than max epoch, no need to train.")
        for epoch in range(current_epoch,max_epoch):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)
            if self.gpu_id == 0:
                logging.info('Saving model Epoch {}'.format(epoch))
                self.save_checkpoint(epoch,self.model_save_path)
            