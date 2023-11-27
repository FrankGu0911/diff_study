import torch

class Atten(torch.nn.Module):
    # single head no mask
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=128,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Linear(128, 128)
        self.k = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, 128)

    def forward(self, x):
        #x -> [1, 128, 32, 32]
        res = x
        #norm,维度不变
        #[1, 128, 32, 32]
        x = self.norm(x)
        #[1, 128, 32, 32] -> [1, 128, 1024] -> [1, 1024, 128]
        x = x.flatten(start_dim=2).transpose(1, 2).contiguous()
        #线性运算,维度不变
        #[1, 1024, 128]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #[1, 1024, 128] -> [1, 128, 1024]
        k = k.transpose(1, 2).contiguous()
        #[1, 1024, 128] * [1, 128, 1024] -> [1, 1024, 1024]
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
        atten = atten.transpose(1, 2).contiguous().reshape(-1, 128, 32, 32)
        #残差连接,维度不变
        #[1, 512, 32, 32]
        atten = atten + res
        return atten
    
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
        # x -> [1, 320, 64, 64]
        # time -> [1, 1280]
        res = x
        # [1, 1280] -> [1, 640, 1, 1]
        time = self.time(time)
        # [1, 320, 64, 64] -> [1, 640, 32, 32]
        x = self.s0(x) + time
        # [1, 640, 32, 32]
        x = self.s1(x)
        # [1, 320, 64, 64] -> [1, 640, 32, 32]
        if self.res:
            res = self.res(res)
        # [1, 640, 32, 32]
        x = res + x
        return x
    
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

class CrossAttention(torch.nn.Module):

    def __init__(self, dim_q, dim_kv):
        #dim_q -> 320
        #dim_kv -> 768

        super().__init__()

        self.dim_q = dim_q

        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        #x -> [1, 4096, 320]
        #kv -> [1, 77, 768]
        #[1, 4096, 320] -> [1, 4096, 320]
        q = self.q(q)
        #[1, 77, 768] -> [1, 77, 320]
        k = self.k(kv)
        #[1, 77, 768] -> [1, 77, 320]
        v = self.v(kv)

        def reshape1(x):
            #x -> [1, 4096, 320]
            b, lens, dim = x.shape
            #[1, 4096, 320] -> [1, 4096, 8, 40]
            x = x.reshape(b, lens, 8, dim // 8)
            #[1, 4096, 8, 40] -> [1, 8, 4096, 40]
            x = x.transpose(1, 2)
            #[1, 8, 4096, 40] -> [8, 4096, 40]
            x = x.reshape(b * 8, lens, dim // 8)
            return x
        #[1, 4096, 320] -> [8, 4096, 40]
        q = reshape1(q)
        #[1, 77, 320] -> [8, 77, 40]
        k = reshape1(k)
        #[1, 77, 320] -> [8, 77, 40]
        v = reshape1(v)

        #[8, 4096, 40] * [8, 40, 77] -> [8, 4096, 77]
        #atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // 8)**-0.5
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device).to(q.dtype),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q // 8)**-0.5,
        )
        atten = atten.softmax(dim=-1)
        #[8, 4096, 77] * [8, 77, 40] -> [8, 4096, 40]
        atten = atten.bmm(v)

        def reshape2(x):
            #x -> [8, 4096, 40]
            b, lens, dim = x.shape
            #[8, 4096, 40] -> [1, 8, 4096, 40]
            x = x.reshape(b // 8, 8, lens, dim)
            #[1, 8, 4096, 40] -> [1, 4096, 8, 40]
            x = x.transpose(1, 2)
            #[1, 4096, 320]
            x = x.reshape(b // 8, lens, dim * 8)
            return x

        #[8, 4096, 40] -> [1, 4096, 320]
        atten = reshape2(atten)
        #[1, 4096, 320] -> [1, 4096, 320]
        atten = self.out(atten)
        return atten

class Transformer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        #in
        self.norm_in = torch.nn.GroupNorm(num_groups=32,
                                          num_channels=dim,
                                          eps=1e-6,
                                          affine=True)
        self.cnn_in = torch.nn.Conv2d(dim,
                                      dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        #atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 768)

        #act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim * 8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim * 4, dim)

        #out
        self.cnn_out = torch.nn.Conv2d(dim,
                                       dim,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, q, kv):
        #q -> [1, 320, 64, 64]
        #kv -> [1, 77, 768]
        b, _, h, w = q.shape
        res1 = q
        #----in----
        #[1, 320, 64, 64]
        q = self.cnn_in(self.norm_in(q))
        #[1, 320, 64, 64] -> [1, 64, 64, 320] -> [1, 4096, 320]
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)
        #----atten----
        #维度不变
        #[1, 4096, 320]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q
        #----act----
        #[1, 4096, 320]
        res2 = q
        #[1, 4096, 320] -> [1, 4096, 2560]
        q = self.fc0(self.norm_act(q))
        #1280
        d = q.shape[2] // 2
        #[1, 4096, 1280] * [1, 4096, 1280] -> [1, 4096, 1280]
        q = q[:, :, :d] * self.act(q[:, :, d:])
        #[1, 4096, 1280] -> [1, 4096, 320]
        q = self.fc1(q) + res2
        #----out----
        #[1, 4096, 320] -> [1, 64, 64, 320] -> [1, 320, 64, 64]
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        #[1, 320, 64, 64]
        q = self.cnn_out(q) + res1
        return q

class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet_time_embed(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet_time_embed(dim_out, dim_out)

        self.out = torch.nn.Conv2d(dim_out,
                                   dim_out,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = []

        out_vae = self.res0(out_vae, time)
        out_vae = self.tf0(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time)
        out_vae = self.tf1(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.out(out_vae)
        outs.append(out_vae)

        return out_vae, outs

class ControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(4, 64, 3, stride=1, padding=1),
                Resnet(64, 64),
                Resnet(64, 64),
                torch.nn.Conv2d(64, 77, 3, stride=1, padding=1),
            )
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)
        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )
        # unet down
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)
        self.down_res0 = Resnet_time_embed(1280, 1280)
        self.down_res1 = Resnet_time_embed(1280, 1280)

        # unet mid
        self.mid_res0 = Resnet_time_embed(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet_time_embed(1280, 1280)

        # condition embed
        self.condition_embed = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1),
        )
        # control net down
        self.control_down = torch.nn.ModuleList([
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])

        # control net mid
        self.control_mid = torch.nn.Conv2d(1280, 1280, kernel_size=1)

    def get_time_embed(self,t):
        #-9.210340371976184 = -math.log(10000)
        # if t.shape == torch.Size([]):
        #     t = t.unsqueeze(0)
        e = torch.arange(160) * -9.210340371976184 / 160
        e = e.exp().to(t.device) 
        e = t.unsqueeze(1) * e.unsqueeze(0)
        #[160+160] -> [320] -> [1, 320]
        e = torch.cat([e.cos(), e.sin()],dim=1)
        return e
    
    def forward(self, out_vae, out_encoder, time, condition):
        # out_vae -> [1, 4, 32, 32]
        # out_encoder -> [1, 4, 768]
        # time -> [1]
        # condition -> [1, 3, 256, 256]

        # in vae [1, 4, 32, 32] -> [1, 320, 32, 32]
        out_vae = self.in_vae(out_vae)
        # in encoder [1, 4, 768] -> [1, 77, 768]
        out_encoder = out_encoder.reshape(-1,4,32,24)
        out_encoder = self.in_encoder(out_encoder).reshape(-1, 77, 768)
        # in time [1] -> [1, 1280]
        time = self.get_time_embed(time)
        if out_vae.dtype == torch.bfloat16:
            time = time.to(torch.bfloat16)
        time = self.in_time(time)
        # condition [1, 3, 256, 256] -> [1, 320, 32, 32]
        condition = self.condition_embed(condition)
        out_vae = out_vae + condition
        # unet down
        out_down = [out_vae] # [1, 320, 32, 32]
        #[1, 320, 32, 32],[1, 77, 768],[1, 1280] -> [1, 320, 32, 32]
        #out_vae -> [1, 320, 16, 16]
        #out -> [1, 320, 32, 32],[1, 320, 32, 32][1, 320, 16, 16]
        out_vae, out = self.down_block0(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)
        #[1, 320, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 16, 16]
        #out_vae -> [1, 640, 8, 8]
        #out -> [1, 640, 16, 16],[1, 640, 16, 16],[1, 640, 8, 8]
        out_vae, out = self.down_block1(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)
        #[1, 640, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 8, 8]
        #out_vae -> [1, 1280, 4, 4]
        #out -> [1, 1280, 8, 8],[1, 1280, 8, 8],[1, 1280, 4, 4]
        out_vae, out = self.down_block2(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)
        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 4, 4]
        out_vae = self.down_res0(out_vae, time)
        out_down.append(out_vae)
        #[1, 1280, 4, 4],[1, 1280] -> [1, 1280, 4, 4]
        out_vae = self.down_res1(out_vae, time)
        out_down.append(out_vae)

        # unet mid
        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res0(out_vae, time)
        #[1, 1280, 8, 8],[1, 77, 768] -> [1, 1280, 8, 8]
        out_vae = self.mid_tf(out_vae, out_encoder)
        #[1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae = self.mid_res1(out_vae, time)

        # control net down
        out_control_down = [self.control_down[i](out_down[i]) for i in range(12)]
        # control net mid
        out_control_mid = self.control_mid(out_vae)

        return out_control_down, out_control_mid

    def load_unet_param(self, unet):
        self.in_encoder.load_state_dict(unet.in_encoder.state_dict())
        self.in_vae.load_state_dict(unet.in_vae.state_dict())
        self.in_time.load_state_dict(unet.in_time.state_dict())
        self.down_block0.load_state_dict(unet.down_block0.state_dict())
        self.down_block1.load_state_dict(unet.down_block1.state_dict())
        self.down_block2.load_state_dict(unet.down_block2.state_dict())
        self.down_res0.load_state_dict(unet.down_res0.state_dict())
        self.down_res1.load_state_dict(unet.down_res1.state_dict())
        self.mid_res0.load_state_dict(unet.mid_res0.state_dict())
        self.mid_tf.load_state_dict(unet.mid_tf.state_dict())
        self.mid_res1.load_state_dict(unet.mid_res1.state_dict())


if __name__ == "__main__":
    controlnet = ControlNet()
    import sys
    sys.path.append('.')
    sys.path.append('./models')
    sys.path.append('./dataset')
    from models.unet import UNet
    unet = UNet().to(torch.bfloat16)
    checkpoint = torch.load('pretrained/diffusion/diffusion_model_5.pth')
    unet.load_state_dict(checkpoint['model_state_dict'])
    controlnet.load_unet_param(unet)
    controlnet = controlnet.to(torch.bfloat16).cuda()
    vae = torch.randn(4,4,32,32).to(torch.bfloat16).cuda()
    encoder = torch.randn(4,4,768).to(torch.bfloat16).cuda()
    condition = torch.randn(4,3,256,256).to(torch.bfloat16).cuda()
    time = torch.LongTensor([0,1,2,3]).cuda()
    out_control_down, out_control_mid = controlnet(vae, encoder, time, condition)
    print("out_control_down:")
    for i in out_control_down:
        print(i.shape)
    print("out_control_mid:")
    print(out_control_mid.shape)
