import torch
import torch.nn as nn

class LCDiff_Planner(nn.Module):
    def __init__(self,pred_len=4):
        super().__init__()
        self.pred_len = pred_len
        self.measurement = nn.Sequential(
                nn.Linear(2+6, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                )
        self.topdown_input = nn.Sequential(
                nn.Linear(4*32*32, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                )
        self.fusion = nn.Sequential(
                nn.Linear(1024+64, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                )
        self.decoder = nn.GRUCell(input_size=4, hidden_size = 128)
        self.output = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 2),
                )
    
    def forward(self, topdown_feature, measurement_feature):
        # topdown_feature: (bs,4,32,32)
        # measurement_feature: (bs,2+6)
        #flatten
        bs = topdown_feature.shape[0]
        topdown_feature = topdown_feature.view(bs,-1)
        measurement_input = self.measurement(measurement_feature)
        topdown_input = self.topdown_input(topdown_feature)
        if torch.isnan(topdown_feature).any():
            print("topdown_feature is nan")
        if torch.isnan(measurement_feature).any():
            print("measurement_feature is nan")
        if torch.isnan(topdown_input).any():
            print("topdown_input is nan")
        if torch.isnan(measurement_input).any():
            print("measurement_input is nan")
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
    model = LCDiff_Planner()
    topdown_feature = torch.randn(4,4,32,32).to(torch.float32)
    measurement_feature = torch.randn(4,2+6).to(torch.float32)
    output = model(topdown_feature, measurement_feature)
    print(output.shape)
