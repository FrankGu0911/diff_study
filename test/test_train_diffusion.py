import torch,sys,os,logging,re
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from diffusers import PNDMScheduler
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from torch.utils.tensorboard import SummaryWriter
TRAIN_NAME = "diffusion"
# path exist
log_path = os.path.join("log",TRAIN_NAME)
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_dir=log_path)
from tqdm import tqdm

from models.vae import VAE
from models.unet import UNet
from models.image_encoder import ImageEncoder
from dataset.carla_dataset import CarlaDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def save_checkpoint(epoch:int,model,opt,path,name=TRAIN_NAME):
    check_point = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(check_point, os.path.join(model_path, f"{name}_{epoch}.pth"))

def latest_model_path(path,name=TRAIN_NAME):
    if not os.path.exists(path):
        return ''
    else:
        re_str = f'{name}_(\d+).pth'
        pat = re.compile(re_str)
        file_list = os.listdir(path)
        epoch_list = []
        for file in file_list:
            res = pat.match(file)
            if res:
                epoch_list.append(int(res.group(1)))
        if len(epoch_list) == 0:
            return ''
        else:
            return os.path.join(path, f"{name}_{max(epoch_list)}.pth")


scheduler = PNDMScheduler(
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

def load_model(model,optimizer,path,model_name,required=False):
    path = os.path.join(path,model_name)
    model_path = latest_model_path(path,model_name)
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        logging.info(f"load model from {model_path}")
    else:
        if required:
            logging.error(f"{model_name} pretrained model not found in {path}")
            raise FileNotFoundError()
        else:
            logging.info(f"{model_name} pretrained model not found in {path}, start from scratch")
            epoch = 0
    return model,optimizer,epoch


if __name__ == '__main__':
    epochs = 5
    batch_size = 1
    dataset = CarlaDataset("test\\data",weathers=[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CarlaDataset.image2topdown_collate_fn)
    vae_model = VAE(26,26).to(device)
    unet_model = UNet().to(device)
    image_encoder = ImageEncoder(77,device).to(device)
    unet_optimizer = torch.optim.AdamW(unet_model.parameters(), lr=1e-5,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
    image_encoder_optimizer = torch.optim.AdamW(image_encoder.parameters(), lr=1e-5,eps=1e-8)
    
    vae_model, _, _ = load_model(vae_model,None,"pretrained","vae_one_hot",required=True)
    unet_model, unet_optimizer, unet_epoch = load_model(unet_model,unet_optimizer,"pretrained","unet")
    image_encoder, image_encoder_optimizer, image_encoder_epoch = load_model(image_encoder,image_encoder_optimizer,"pretrained","image_encoder")
    
    vae_model.requires_grad_(False)
    unet_model.requires_grad_(True)
    image_encoder.requires_grad_(True)

    vae_model.eval()
    unet_model.train()
    image_encoder.train()

    cur_epoch = min(unet_epoch,image_encoder_epoch)
    criterion = torch.nn.MSELoss()

    for e in range(cur_epoch,epochs):
        train_loop = tqdm(dataloader,desc="Train Epoch %d" %e, total=len(dataloader))
        for data in train_loop:
            image, topdown = data
            image = image.to(device)
            topdown = topdown.to(device)
            with torch.no_grad():
                z = vae_model.encoder(topdown)
                z = vae_model.sample(z[0],z[1])
                z = z.detach()
            unet_optimizer.zero_grad()
            image_encoder_optimizer.zero_grad()
            noise = torch.randn_like(z)
            noise_step = torch.randint(0, 1000, (1, )).long().to(device)
            z_noise = scheduler.add_noise(z, noise, noise_step)
            train_loss_list = []
            scaler = GradScaler()
            with autocast():
                out_encoder = image_encoder(image)
                pred = unet_model(z_noise,out_encoder,noise_step)
                loss = criterion(pred,z)
                if torch.isnan(loss):
                    tqdm.write(f"Epoch {e}:\t loss NAN")
                else:
                    train_loss_list.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(unet_optimizer)
            scaler.step(image_encoder_optimizer)
            scaler.update()
            train_loop.set_postfix(loss=np.mean(train_loss_list))
        save_checkpoint(e,unet_model,unet_optimizer,"pretrained","unet")
        save_checkpoint(e,image_encoder,image_encoder_optimizer,"pretrained","image_encoder")

