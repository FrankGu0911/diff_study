import torch,sys,os,logging,re
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
# print(sys.path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='log/vae')
from tqdm import tqdm

from models.vae import VAE
from dataset.carla_topdown_dataset import CarlaTopDownDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def vae_train(cur_epoch,vae_model, vae_opt, loader,epoch):
    vae_model.train()
    for e in range(cur_epoch, epoch):
        scalar_name = 'epoch_%d' % e
        for i, (x, _) in tqdm(enumerate(loader), total = len(loader)):
            x = x.to(device)
            vae_opt.zero_grad()
            con_x, mu, logvar = vae_model(x)
            loss = vae_loss(x, con_x, mu, logvar)
            loss.backward()
            vae_opt.step()
            if i % 10 == 0:
                # print(f"Epoch {e}:{i}\t loss: {loss.item()}")
                writer.add_scalar(scalar_name, loss.item(), i)
        save_checkpoint(e, vae_model, vae_opt, 'pretrained/vae_model')
            
if __name__ == "__main__":
    epoch = 50
    batch_size = 6
    vae_model = VAE(1,1).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    ds = CarlaTopDownDataset('..\\..\\dataset')
    model_path = latest_model_path('pretrained/vae_model')
    if model_path:
        checkpoint = torch.load(model_path)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch'] + 1
    else:
        cur_epoch = 0
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    vae_train(cur_epoch,vae_model, optimizer, loader, epoch)