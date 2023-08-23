import torch,os,sys,logging,re,argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.unet import UNet
from dataset.carla_dataset import CarlaDataset
from trainer.DiffusionTrainer import DiffusionTrainer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def SetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true",default=False)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--epoch",type=int,default=35)
    parser.add_argument("--autocast",action="store_true",default=False)
    parser.add_argument("--lidar",action="store_true",default=False)
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
        pat = re.compile(r'diffusion_model_(\d+).pth')
        file_list = os.listdir(path)
        epoch_list = []
        for file in file_list:
            res = pat.match(file)
            if res:
                epoch_list.append(int(res.group(1)))
        if len(epoch_list) == 0:
            return ''
        else:
            return os.path.join(path, f"diffusion_model_{max(epoch_list)}.pth")

if __name__ == "__main__":
    ddp_setup()
    args = SetArgs()
    device = torch.device("cuda:%d" % int(os.environ["LOCAL_RANK"]))
    unet_model = UNet(args.lidar).to(device)
    unet_optimizer = torch.optim.AdamW(unet_model.parameters(),lr=5e-5,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(unet_optimizer,T_0=2,T_mult=2,eta_min=1e-6)
    train_ds = CarlaDataset('E:/dataset',weathers=[0,1,2,3,4,5,6,7,8,9,10],towns=[1,2,3,4,5,6,7,10],topdown_base_weight=1,topdown_diff_weight=100)
    val_ds = CarlaDataset('E:/dataset',weathers=[11,12,13],towns=[1,2,3],topdown_base_weight=1,topdown_diff_weight=100)
    if args.lidar:
        train_loader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=CarlaDataset.clip_lidar_feature2vae_feature_collate_fn,
                                sampler=DistributedSampler(train_ds)
                                )
        val_loader = DataLoader(val_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=CarlaDataset.clip_lidar_feature2vae_feature_collate_fn,
                                sampler=DistributedSampler(val_ds)
                                )
    else:
        train_loader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=CarlaDataset.clip_feature2vae_feature_collate_fn,
                                sampler=DistributedSampler(train_ds)
                                )
        val_loader = DataLoader(val_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=CarlaDataset.clip_feature2vae_feature_collate_fn,
                                sampler=DistributedSampler(val_ds)
                                )
    if args.lidar:
        model_path = os.path.join("pretrained",'diffusion_lidar')
    else:
        model_path = os.path.join("pretrained",'diffusion')
    CheckPath(model_path)
    if args.resume:
        model_param = latest_model_path(model_path)
    else:
        model_param = ''
    if model_param:
        checkpoint = torch.load(model_param,map_location=device)
        current_epoch = checkpoint["epoch"] + 1
        scheduler.last_epoch = checkpoint["epoch"]
        unet_model.load_state_dict(checkpoint["model_state_dict"])
        unet_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = None
        torch.cuda.empty_cache()
    else:
        current_epoch = 0
    if os.environ["LOCAL_RANK"] == "0":
        logging.info(f"Start at epoch{current_epoch}")
        log_path = os.path.join("log",'diffusion')
        CheckPath(log_path)
        writer = SummaryWriter(log_dir=log_path)
    else:
        writer = None
    trainer = DiffusionTrainer(unet_model=unet_model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                optimizer=unet_optimizer,
                                lr_scheduler=scheduler,
                                autocast=args.autocast,
                                writer=writer,
                                model_save_path=model_path,
                                dist=True,
                                with_lidar=args.lidar)
    trainer.train(current_epoch,max_epoch=args.epoch)
    destroy_process_group()