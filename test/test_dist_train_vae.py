import torch,os,sys,logging,re,argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.vae import VAE
from dataset.carla_topdown_dataset import CarlaTopDownDataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def SetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_name",type=str,default="vae_one_hot")
    parser.add_argument("--resume",action="store_true",default=False)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--epoch",type=int,default=35)
    parser.add_argument("--autocast",action="store_true",default=False)
    return parser.parse_args()

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

def vae_loss(ori_x, con_x, mu, logvar):
    # the loss from the reconstruct image -> actually cannot describe the quality of the whole image
    bce_loss = torch.nn.functional.mse_loss(con_x.view(-1), ori_x.view(-1), reduction='sum')
    # how close the two distributions are (x and normal distribution)
    kl_diverage = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_diverage

def ddp_setup():
    # initialize the process group
    init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def latest_model_path(path):
    if not os.path.exists(path):
        return ''
    else:
        pat = re.compile(r'vae_model_(\d+).pth')
        file_list = os.listdir(path)
        epoch_list = []
        for file in file_list:
            res = pat.match(file)
            if res:
                epoch_list.append(int(res.group(1)))
        if len(epoch_list) == 0:
            return ''
        else:
            return os.path.join(path, f"vae_model_{max(epoch_list)}.pth")

class VAETrainer:
    def __init__(self,
                 vae_model:torch.nn.Module,
                 train_loader:torch.utils.data.DataLoader,
                 val_loader:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 autocast:bool=False,
                 writer:SummaryWriter=None,
                 model_save_path:str=None):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = vae_model.cuda(self.gpu_id)
        self.optimizer = optimizer
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
        for (x,_) in train_loop:
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
            
if __name__ == "__main__":
    ddp_setup()
    args = SetArgs()
    device = torch.device("cuda:%d" % int(os.environ["LOCAL_RANK"]))
    vae_model = VAE(26,26).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4)
    train_ds = CarlaTopDownDataset('../dataset-remote',onehot=True,weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,3,4,5,6,7,10],base_weight=1,diff_weight=100)
    val_ds = CarlaTopDownDataset('../dataset-remote',onehot=True,weathers=[11,12,13],towns=[1,2,3],base_weight=1,diff_weight=100)
    model_path = os.path.join("pretrained",args.train_name)
    CheckPath(model_path)
    if args.resume:
        model_param = latest_model_path(model_path)
    if model_param:
        checkpoint = torch.load(model_param,map_location=device)
        current_epoch = checkpoint["epoch"] + 1
        vae_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        current_epoch = 0
    if os.environ["LOCAL_RANK"] == "0":
        logging.info(f"Start at epoch{current_epoch}")
        log_path = os.path.join("log",args.train_name)
        CheckPath(log_path)
        writer = SummaryWriter(log_dir=log_path)
    else:
        writer = None
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True,
                              sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            sampler=DistributedSampler(val_ds))
    trainer = VAETrainer(vae_model=vae_model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            autocast=args.autocast,
                            writer=writer,
                            model_save_path=model_path)
    trainer.train(current_epoch,max_epoch=args.epoch)
    destroy_process_group()
