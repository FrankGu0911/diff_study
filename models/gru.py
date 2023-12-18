import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self,pred_len=4,with_rgb=False,with_lidar=False):
        super().__init__()
        self.pred_len = pred_len
        self.with_rgb = with_rgb
        self.with_lidar = with_lidar
        self.measurement = nn.Sequential(
                nn.Linear(2+6, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 256),
                nn.LeakyReLU(),
                )
        self.topdown_input = nn.Sequential(
                nn.Linear(4*32*32, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 256),
                nn.LeakyReLU(),
                )
        if self.with_rgb:
            self.rgb_input = nn.Sequential(
                    nn.Linear(4*768, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 256),
                    nn.LeakyReLU(),
                    )
        if self.with_lidar:
            self.lidar_input = nn.Sequential(
                    nn.Linear(256*256, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 256),
                    nn.LeakyReLU(),
                    )
        if self.with_rgb and self.with_lidar:
            self.fusion = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    )
        elif self.with_rgb:
            self.fusion = nn.Sequential(
                    nn.Linear(512+256, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    )
        elif self.with_lidar:
            self.fusion = nn.Sequential(
                    nn.Linear(512+256, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    )
        else:
            self.fusion = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    )
        self.decoder = nn.GRUCell(input_size=4, hidden_size = 512)
        self.output = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 2),
                )
    
    def forward(self, topdown_feature, measurement_feature,rgb_feature=None,lidar_feature=None):
        # topdown_feature: (bs,4,32,32)
        # measurement_feature: (bs,2+6)
        # rgb_feature: (bs,4,768)
        # lidar_feature: (bs,1,256,256)
        #flatten
        bs = topdown_feature.shape[0]
        topdown_feature = topdown_feature.view(bs,-1)
        measurement_input = self.measurement(measurement_feature)
        topdown_input = self.topdown_input(topdown_feature)
        if self.with_rgb:
            rgb_feature = rgb_feature.view(bs,-1)
            rgb_input = self.rgb_input(rgb_feature)
        if self.with_lidar:
            lidar_feature = lidar_feature.view(bs,-1)
            lidar_input = self.lidar_input(lidar_feature)
        if self.with_rgb and self.with_lidar:
            z = self.fusion(torch.cat([topdown_input, rgb_input, lidar_input, measurement_input], dim=1))
        elif self.with_rgb:
            z = self.fusion(torch.cat([topdown_input, rgb_input, measurement_input], dim=1))
        elif self.with_lidar:
            z = self.fusion(torch.cat([topdown_input, lidar_input, measurement_input], dim=1))
        else:
            z = self.fusion(torch.cat([topdown_input, measurement_input], dim=1))
        # print(topdown_input.shape)
        # print(measurement_input.shape)
        x  = torch.zeros(bs,2).to(measurement_feature.dtype).to(measurement_feature.device)
        # print(z.shape)
        target_point = measurement_feature[:,:2]
        output = []
        for _ in range(self.pred_len):
            x_in = torch.cat((x,target_point),dim=1)
            z = self.decoder(x_in,z)
            dx = self.output(z)
            x = x + dx
            output.append(x)
        output = torch.stack(output,dim=1)
        return output

if __name__ == "__main__":
    model = GRU(with_lidar=False,with_rgb=True)
    topdown_feature = torch.randn(4,4,32,32).to(torch.float32)
    measurement_feature = torch.randn(4,2+6).to(torch.float32)
    rgb_feature = torch.randn(4,4,768).to(torch.float32)
    lidar_feature = torch.randn(4,256,256).to(torch.float32)
    output = model(topdown_feature, measurement_feature,rgb_feature=rgb_feature)
    print(output.shape)
