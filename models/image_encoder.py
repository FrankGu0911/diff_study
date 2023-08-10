import torch,sys
from torch.cuda.amp import autocast
import clip
from PIL import Image
from torchvision import transforms

class Resnet(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )
        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
    def forward(self, x):
        #x -> [1, dim_in, resx, resy]
        res = x
        if self.res:
            #[1, dim_in, resx, resy] -> [1, dim_in, resx, resy]
            res = self.res(x)
        #[1, dim_in, resx, resy] -> [1, dim_in, resx, resy]
        return res + self.s(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self, 
                 latent_dim, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.clip_encoder, _ = clip.load('ViT-L/14', device=device)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, 3, stride=1, padding=1),
            Resnet(64, 64),
            Resnet(64, 64),
            torch.nn.Conv2d(64, self.latent_dim, 3, stride=1, padding=1),
        ).to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])

    def forward(self, image:torch.Tensor):
        # image -> [bs, 4, 3, 600, 800]
        # get the bs
        image = image.reshape(-1, 3, 600, 800)
        image = self.preprocess(image.to(self.device))
        features = self.clip_encoder.encode_image(image)
        features = features.detach().float().reshape(-1,4, 32, 24)
        features = self.head(features).reshape(-1, self.latent_dim, 32*24)
        return features
    

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    sys.path.append('./dataset')
    from dataset.carla_dataset import CarlaDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_encoder = ImageEncoder(77, device)
    dataset = CarlaDataset("E:\\dataset")
    image_full = torch.cat((dataset[0][0].image_full.unsqueeze(0),dataset[0][0].image_full.unsqueeze(0)),dim=0)
    print(image_full.shape)
    features = image_encoder(image_full)
    print(features.shape)
    