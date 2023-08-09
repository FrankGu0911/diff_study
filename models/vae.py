import torch
from torch.cuda.amp import autocast
class Pad(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.pad(x, (0, 1, 0, 1),
                                       mode='constant',
                                       value=0)

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


class Atten(torch.nn.Module):
    # single head no mask
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        #x -> [1, 512, 32, 32]
        res = x
        #norm,维度不变
        #[1, 512, 32, 32]
        x = self.norm(x)
        #[1, 512, 32, 32] -> [1, 512, 1024] -> [1, 1024, 512]
        x = x.flatten(start_dim=2).transpose(1, 2).contiguous()
        #线性运算,维度不变
        #[1, 1024, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #[1, 1024, 512] -> [1, 512, 1024]
        k = k.transpose(1, 2).contiguous()
        #[1, 1024, 512] * [1, 512, 1024] -> [1, 1024, 1024]
        #0.044194173824159216 = 1 / 512**0.5
        atten = q.bmm(k) * 0.044194173824159216
        #照理来说应该是等价的,但是却有很小的误差
        # atten = torch.baddbmm(torch.empty(1, 1024, 1024, device=q.device),
        #                       q,
        #                       k,
        #                       beta=0,
        #                       alpha=0.044194173824159216)
        atten = torch.softmax(atten, dim=2)
        #[1, 1024, 1024] * [1, 1024, 512] -> [1, 1024, 512]
        atten = atten.bmm(v)
        #线性运算,维度不变
        #[1, 1024, 512]
        atten = self.out(atten)
        #[1, 1024, 512] -> [1, 512, 1024] -> [1, 512, 32, 32]
        atten = atten.transpose(1, 2).contiguous().reshape(-1, 512, 32, 32)
        #残差连接,维度不变
        #[1, 512, 32, 32]
        atten = atten + res
        return atten

class Encoder(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            #in 
            torch.nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),

            #down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            #mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            #正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )
    def forward(self, x):
        h = self.encoder(x)
        return h[:, :4], h[:, 4:] # mean, logvar

class Decoder(torch.nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4, 4, 1),
            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),
            #middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),
            #up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),
            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
            ),
        )
    def forward(self, x):
        return self.decoder(x)

class VAE(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(output_channels)
    
    def sample(self, mean, logvar):
        #[1, 4, 32, 32]
        std = logvar.exp()**0.5
        #[1, 4, 32, 32]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h
        return h
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        x = self.sample(mean, logvar)
        x = self.decoder(x)
        return x, mean, logvar

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(26, 26).to(device)
    x = torch.randn(1, 26, 256, 256).to(device)
    # y = model(x)
    # print(y[0].shape, y[1].shape, y[2].shape)
    mean, logvar = model.encoder(x)
    y = model.sample(mean, logvar)
    print(y.shape)