import torch,sys,os,logging,re
from torch.cuda.amp import autocast, GradScaler
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
# print(sys.path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from torch.utils.tensorboard import SummaryWriter
TRAIN_NAME = "vae_one_hot"
# path exist
log_path = os.path.join("log",TRAIN_NAME)
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_dir=log_path)
from tqdm import tqdm

from models.vae import VAE
from dataset.carla_topdown_dataset import CarlaTopDownDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def save_checkpoint(epoch:int,model,opt,path):
    check_point = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(check_point, os.path.join(path, f"vae_model_{epoch}.pth"))

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

def vae_loss(ori_x, con_x, mu, logvar):
    # the loss from the reconstruct image -> actually cannot describe the quality of the whole image
    bce_loss = torch.nn.functional.mse_loss(con_x.view(-1), ori_x.view(-1), reduction='sum')
    # how close the two distributions are (x and normal distribution)
    kl_diverage = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_diverage

def vae_train(cur_epoch,vae_model,vae_opt,train_loader,val_loader,epoch):
    for e in range(cur_epoch, epoch):
        # scalar_name = 'epoch_%d' % e
        vae_model.train()
        train_loop = tqdm(train_loader,desc="Train Epoch %d" %e, total=len(train_loader))
        train_loss_list = []
        scaler = GradScaler()
        for (x, _) in train_loop:
            x = x.to(device)
            vae_opt.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            with autocast():
                con_x, mu, logvar = vae_model(x)
                loss = vae_loss(x, con_x, mu, logvar)
            if torch.isnan(loss):
                tqdm.write(f"Epoch {e}:\t loss NAN")
            else:
                train_loss_list.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(vae_opt)
            scaler.update()
            # loss.backward()
            # vae_opt.step()
            # tqdm.write(f"Epoch {e}:\t loss: {loss.item()}")
            

        writer.add_scalar('train_loss', sum(train_loss_list)/len(train_loss_list), e)
        vae_model.eval()
        val_loop = tqdm(val_loader,desc="Valid Epoch %d" %e, total=len(val_loader))
        val_loss_list = []
        for (x, _) in val_loop:
            with torch.no_grad():
                x = x.to(device)
                con_x, mu, logvar = vae_model(x)
                loss = vae_loss(x, con_x, mu, logvar)
                if torch.isnan(loss):
                    tqdm.write(f"Epoch {e}:\t loss NAN")
                else:
                    val_loss_list.append(loss.item())
        writer.add_scalar('val_loss', sum(val_loss_list)/len(val_loss_list), e)
        model_path = os.path.join("pretrained",TRAIN_NAME)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_checkpoint(e, vae_model, vae_opt, model_path)
            
if __name__ == "__main__":
    epoch = 30
    batch_size = 1
    vae_model = VAE(26,26).to(device)
    # optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-4)
    #adamw
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=5e-5)
    train_ds = CarlaTopDownDataset('test/data',onehot=True,weathers=[0,1,2,3,4,5,6,7,8,9,10],base_weight=1,diff_weight=100)
    val_ds = CarlaTopDownDataset('test/data',onehot=True,weathers=[11,12,13],base_weight=1,diff_weight=100)
    # train_ds = CarlaTopDownDataset('/home/frank/code/dataset',onehot=True,weathers=[0,1,2,3,4,5,6,7,8,9,10],base_weight=0.1,diff_weight=2)
    # val_ds = CarlaTopDownDataset('/home/frank/code/dataset',onehot=True,weathers=[11,12,13],base_weight=0.1,diff_weight=2)
    model_path = os.path.join("pretrained",TRAIN_NAME)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = latest_model_path(model_path)
    if model:
        checkpoint = torch.load(model)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch'] + 1
        checkpoint = None
        torch.cuda.empty_cache()
    else:
        cur_epoch = 0
    # cur_epoch = 0
    logging.info(f"Start at epoch{cur_epoch}")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    if cur_epoch < epoch:
        vae_train(cur_epoch,vae_model, optimizer, train_loader,val_loader, epoch)
    else:
        pass
    # os.system('shutdown -s -t 60')