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
from trainer.VAETrainer import VAETrainer
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

if __name__ == "__main__":
    ddp_setup()
    args = SetArgs()
    device = torch.device("cuda:%d" % int(os.environ["LOCAL_RANK"]))
    vae_model = VAE(26,26).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2,eta_min=1e-6)
    train_ds = CarlaTopDownDataset('../dataset-remote/dataset',onehot=True,weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],towns=[1,2,3,4,5,6,7,10],base_weight=1,diff_weight=100)
    val_ds = CarlaTopDownDataset('../dataset-remote/dataset',onehot=True,weathers=[11,15,17],towns=[1,2,3],base_weight=1,diff_weight=100)
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
                            lr_scheduler=scheduler,
                            autocast=args.autocast,
                            writer=writer,
                            model_save_path=model_path)
    trainer.train(current_epoch,max_epoch=args.epoch)
    destroy_process_group()
