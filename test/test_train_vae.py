import torch,sys,os
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')

# print(sys.path)

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
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            vae_opt.zero_grad()
            con_x, mu, logvar = vae_model(x)
            loss = vae_loss(x, con_x, mu, logvar)
            loss.backward()
            vae_opt.step()
            if i % 10 == 0:
                print(f"Epoch {e}:{i}\t loss: {loss.item()}")
        torch.save(vae_model.state_dict(), f"vae_model_{e}.pth")
            
if __name__ == "__main__":
    epoch = 50
    batch_size = 2
    vae_model = VAE(1,1).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    ds = CarlaTopDownDataset('test/data')
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    vae_train(vae_model, optimizer, loader, epoch)