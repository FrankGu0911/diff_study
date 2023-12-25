import torch,os,sys,logging,re,argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.gru import GRU
from dataset.carla_dataset import CarlaDataset
from trainer.LCDiffPlannerTrainer import LCDiffPlannerTrainer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def SetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',"--resume",action="store_true",default=False)
    parser.add_argument('-b',"--batch_size",type=int,default=8)
    parser.add_argument('-e',"--epoch",type=int,default=35)
    parser.add_argument('-l',"--pred_len",type=int,default=4)
    parser.add_argument('-rgb',"--with_rgb",action="store_true",default=False)
    parser.add_argument("-lidar","--with_lidar",action="store_true",default=False)
    parser.add_argument('-sr',"--with_stop_reason",action="store_true",default=False)
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
        pat = re.compile(r'gru_model_(\d+).pth')
        file_list = os.listdir(path)
        epoch_list = []
        for file in file_list:
            res = pat.match(file)
            if res:
                epoch_list.append(int(res.group(1)))
        if len(epoch_list) == 0:
            return ''
        else:
            return os.path.join(path, f"gru_model_{max(epoch_list)}.pth")
    
if __name__ == "__main__":
    args = SetArgs()
    device = torch.device("cuda:0")
    gru_model = GRU(with_lidar=args.with_lidar,with_rgb=args.with_rgb,with_stop_reason=args.with_stop_reason)
    if args.half:
        gru_model = gru_model.to(torch.bfloat16)
    gru_model = gru_model.to(device)
    gru_optimizer = torch.optim.AdamW(gru_model.parameters(),lr=2e-4,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(gru_optimizer,T_0=25,T_mult=2,eta_min=1e-6)
    train_ds = CarlaDataset('E:\\remote\\dataset-full',weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,3,4,5,6,7,10],seq_len=1,pred_len=args.pred_len)
    val_ds = CarlaDataset('E:\\remote\\dataset-val',weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,4,5,6,7,10],seq_len=1,pred_len=args.pred_len)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=CarlaDataset.vae_clip_lidar_measurement2cmdwp_collate_fn,
                              num_workers=12,
                              pin_memory=True,
                              )
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=CarlaDataset.vae_clip_lidar_measurement2cmdwp_collate_fn,
                            num_workers=12,
                            pin_memory=True
                            )
    model_name = 'gru'
    if args.with_rgb:
        model_name+='_rgb'
    if args.with_lidar:
        model_name+='_lidar'
    if args.with_stop_reason:
        model_name+='_reason'
    model_path =os.path.join('pretrained',model_name)
    CheckPath(model_path)
    if args.resume:
        model_param = latest_model_path(model_path)
    else:
        model_param = ''
    if model_param:
        checkpoint = torch.load(model_param,map_location=device)
        current_epoch = checkpoint["epoch"] + 1
        scheduler.last_epoch = checkpoint["epoch"]
        gru_model.load_state_dict(checkpoint["model_state_dict"])
        gru_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = None
        torch.cuda.empty_cache()
    else:
        current_epoch = 0
    logging.info(f"Start at epoch{current_epoch}")
    log_path = os.path.join('log',model_name)
    CheckPath(log_path)
    writer = SummaryWriter(log_dir=log_path)
    trainer = LCDiffPlannerTrainer(gru_model=gru_model,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    optimizer=gru_optimizer,
                                    lr_scheduler=scheduler,
                                    autocast=args.autocast,
                                    with_lidar=args.with_lidar,
                                    with_rgb=args.with_rgb,
                                    with_stop_reason=args.with_stop_reason,
                                    writer=writer,
                                    model_save_path=model_path,
                                    dist=False,
                                    half=args.half)
    trainer.train(current_epoch,args.epoch)
