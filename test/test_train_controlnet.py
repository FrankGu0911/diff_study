import torch,os,sys,logging,re,argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.unet import UNet
from models.controlnet import ControlNet
from dataset.carla_dataset import CarlaDataset
from trainer.ControlNetTrainer import ControlNetTrainer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def SetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true",default=False)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--epoch",type=int,default=35)
    parser.add_argument("--autocast",action="store_true",default=False)
    parser.add_argument("--half",action="store_true",default=False)
    return parser.parse_args()

def CheckPath(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

def latest_model_path(path):
    if not os.path.exists(path):
        return ''
    else:
        pat = re.compile(r'controlnet_(\d+).pth')
        file_list = os.listdir(path)
        epoch_list = []
        for file in file_list:
            res = pat.match(file)
            if res:
                epoch_list.append(int(res.group(1)))
        if len(epoch_list) == 0:
            return ''
        else:
            return os.path.join(path, f"controlnet_{max(epoch_list)}.pth")

if __name__ == "__main__":
    args = SetArgs()
    device = torch.device("cuda:0")
    unet_model_path = "pretrained/diffusion/diffusion_model_36.pth"
    unet_model = UNet()
    controlnet_model = ControlNet()
    if args.half:
        unet_model = unet_model.to(torch.bfloat16)
        controlnet_model = controlnet_model.to(torch.bfloat16)
    unet_model = unet_model.to(device)
    controlnet_model = controlnet_model.to(device)
    controlnet_optimizer = torch.optim.AdamW(controlnet_model.parameters(),lr=1e-4,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(controlnet_optimizer,T_0=args.epoch,T_mult=2,eta_min=1e-6)
    train_ds = CarlaDataset('/root/autodl-tmp/remote/dataset-full',weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,3,4,5,6,7,10],interval_frame=4)
    # train_ds = CarlaDataset('/data/zjw/frank/dataset-remote/dataset-full',weathers=[4],towns=[1])
    val_ds = CarlaDataset('/root/autodl-tmp/remote/dataset-val',weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,3,4,5,6,7,10],interval_frame=2)
    train_loader = DataLoader(train_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=CarlaDataset.clip_lidar2d_feature2vae_feature_collate_fn,
                            pin_memory=True,
                            num_workers=16,
                            )
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=CarlaDataset.clip_lidar2d_feature2vae_feature_collate_fn,
                            pin_memory=True,
                            num_workers=16,
                            )
    unet_model.load_state_dict(torch.load(unet_model_path,map_location=device)["model_state_dict"])
    model_path = os.path.join("pretrained",'controlnet')
    CheckPath(model_path)
    if args.resume:
        model_param = latest_model_path(model_path)
    else:
        model_param = ''
    if model_param:
        checkpoint = torch.load(model_param,map_location=device)
        current_epoch = checkpoint["epoch"] + 1
        scheduler.last_epoch = checkpoint["epoch"]
        controlnet_model.load_state_dict(checkpoint["model_state_dict"])
        controlnet_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = None
        resume = True
        torch.cuda.empty_cache()
    else:
        current_epoch = 0
        resume = False
    if not args.half:
        unet_model = unet_model.to(torch.float32)
        controlnet_model = controlnet_model.to(torch.float32)
    logging.info(f"Start at epoch{current_epoch}")
    log_path = os.path.join("log",'controlnet')
    CheckPath(log_path)
    writer = SummaryWriter(log_dir=log_path)
    trainer = ControlNetTrainer(unet_model=unet_model,
                                controlnet_model=controlnet_model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                optimizer=controlnet_optimizer,
                                lr_scheduler=scheduler,
                                autocast=args.autocast,
                                writer=writer,
                                model_save_path=model_path,
                                dist=False,
                                half=args.half,
                                resume=resume)
    trainer.train(current_epoch,max_epoch=args.epoch)