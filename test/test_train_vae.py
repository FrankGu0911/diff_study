import torch,sys,os,logging
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
def vae_loss(ori_x, con_x, mu, logvar):
    # the loss from the reconstruct image -> actually cannot describe the quality of the whole image
    bce_loss = torch.nn.functional.mse_loss(con_x.view(-1), ori_x.view(-1), reduction='sum')
    # how close the two distributions are (x and normal distribution)
    kl_diverage = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_diverage

def vae_train(vae_model, vae_opt, loader,epoch):
    vae_model.train()
    for e in range(epoch):
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
        torch.save(vae_model.state_dict(), f"vae_model_{e}.pth")
            
if __name__ == "__main__":
    epoch = 50
    batch_size = 8
    vae_model = VAE(1,1).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    ds = CarlaTopDownDataset('..\\..\\dataset')
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    vae_train(vae_model, optimizer, loader, epoch)