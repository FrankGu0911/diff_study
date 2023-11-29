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
        self.decoder = nn.GRUCell(input_size=4, hidden_size = 1024+64)
        self.output = nn.Sequential(
                nn.Linear(1024+64, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                )
    
    def forward(self, topdown_feature, measurement_feature):
        # topdown_feature: (bs,4,32,32)
        # measurement_feature: (bs,2+6)
        #flatten
        bs = topdown_feature.shape[0]
        topdown_feature = topdown_feature.view(bs,-1)
        measurement_input = self.measurement(measurement_feature)
        topdown_input = self.topdown_input(topdown_feature)
        print(topdown_input.shape)
        print(measurement_input.shape)
        z = torch.cat([topdown_input, measurement_input], dim=1)
        x  = torch.zeros(bs,2).to(measurement_feature.dtype)
        print(z.shape)
        target_point = measurement_feature[:,:2]
        output = []
        for _ in range(self.pred_len):
            x_in = torch.cat((x,target_point),dim=1)
            z = self.decoder(x_in,z)
            x = self.output(z)
            output.append(x)
        output = torch.stack(output,dim=1)
        return output

if __name__ == "__main__":
    model = LCDiff_Planner()
    topdown_feature = torch.randn(4,4,32,32).to(torch.float32)
    measurement_feature = torch.randn(4,2+6).to(torch.float32)
    output = model(topdown_feature, measurement_feature)
    print(output.shape)
