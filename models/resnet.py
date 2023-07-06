import torch

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

class Resnet_time_embed(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )
        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_in,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_in,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )
        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_out,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_out,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )
        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv2d(dim_in,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

    def forward(self, x, time):
        #x -> [1, 320, 64, 64]
        #time -> [1, 1280]
        res = x
        #[1, 1280] -> [1, 640, 1, 1]
        time = self.time(time)
        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        x = self.s0(x) + time
        #[1, 640, 32, 32]
        x = self.s1(x)
        #[1, 320, 64, 64] -> [1, 640, 32, 32]
        if self.res:
            res = self.res(res)
        #[1, 640, 32, 32]
        x = res + x
        return x

