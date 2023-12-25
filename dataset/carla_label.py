import os,re,logging,torch,json
import numpy as np
from PIL import Image
from torchvision import transforms
_logger = logging.getLogger(__name__)

class CarlaLabel():
    def __init__(
        self,
        path: str,
        index: int,
        important_seg: list = [4,19,23],
        base_weight: int = 1,
        diff_weight: int = 100,
        pred_len: int = 1,
        gen_feature: bool = True,
        vae_model_path: str = None,
        cache = None
        ):
        self.root_path = path
        self.relative_path = "/".join(self.root_path.replace("\\","/").split("/")[-3:])
        self.index = index
        self.pred_len = pred_len
        self.cache = cache
        self._topdown = None
        self._topdown_onehot = None
        self._vae_feature = None
        self._measurements = None
        self.important_seg = important_seg
        self.base_weight = base_weight
        self.diff_weight = diff_weight
        self.gen_feature = gen_feature
        self.vae_model_path = vae_model_path
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __repr__(self) -> str:
        return "Label: %d at %s" % (self.index, self.root_path)
    
    def __str__(self) -> str:
        return "Label: %d at %s" % (self.index, self.root_path)
    
    def calc_crop(self,tar_x, tar_y):
        if tar_x > 512 or tar_y > 512:
            return (0,0,512,512)
        if tar_x < 0 or tar_y < 0:
            raise ValueError("Target size should be positive")
        if tar_x > 256 or tar_y > 256:
            x = (512 - tar_x) // 2
            return (x,0,x+tar_x,tar_y)
        else:
            x = (512 - tar_x) // 2
            y = 256 - tar_y
            return (x,y,x+tar_x,y+tar_y)

    def _LoadImage(self, name: str, idx: int, ext: str='jpg'):
        path = os.path.join(self.root_path, name, "%04d.%s" % (idx, ext))
        if not os.path.exists(path):
            logging.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")
        logging.debug(f"Load image from {path}")
        return Image.open(path)
    
    def _LoadJson(self, name: str, idx: int):
        path = os.path.join(self.root_path, name, "%04d.json" % idx)
        if not os.path.exists(path):
            logging.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")
        logging.debug(f"Load json from {path}")
        return json.load(open(path, "r"))

    def get_one_hot(self,label, N):
        dtype = label.dtype
        shape = label.shape
        ones = torch.eye(N)
        ones = ones * self.base_weight
        important = self.important_seg
        for i in important:
            ones[i][i] = self.diff_weight
        ones[6][6] = max(self.base_weight,self.diff_weight/10)
        onehot = ones.index_select(0, label.int().view(-1)).reshape(*shape, N).to(dtype).squeeze(0).permute(2,0,1)
        return onehot

    @property
    def topdown(self):
        if self._topdown is None:
            self._topdown = self._LoadImage("topdown", self.index, "png").crop(self.calc_crop(256,256))
        topdown = (transforms.ToTensor()(self._topdown) * 255).to(torch.uint8)
        if torch.max(topdown) > 25:
            _logger.debug("Topdown image has value larger than 25: %s %s" % (self.root_path, self.index))
            # replace with 7
            topdown = torch.where(topdown > 25, torch.Tensor([7]).to(torch.uint8), topdown)
        return topdown
    
    @property
    def topdown_onehot(self):
        if self._topdown_onehot is None:
            self._topdown_onehot = self.get_one_hot(self.topdown, 26).to(torch.float32)
        return self._topdown_onehot
    
    @property
    def vae_feature(self):
        if self._vae_feature is None:
            if os.path.exists(os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index)):
                self._vae_feature = torch.load(os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index),map_location='cpu')
            else:
                logging.debug("Vae Feature %s does not exist" % os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index))
                if self.gen_feature:
                    if self.vae_model_path is None:
                        logging.error('Choose to Generate VAE Feature but cannot find model path')
                        raise FileNotFoundError('Choose to Generate VAE Feature but cannot find model path')
                    import sys
                    sys.path.append('.')
                    from models.vae import VAE
                    vae = VAE(26,26).to('cuda' if torch.cuda.is_available() else 'cpu')
                    vae.load_state_dict(torch.load(self.vae_model_path)['model_state_dict'])
                    vae.eval()
                    with torch.no_grad():
                        mean, logvar = vae.encoder(self.topdown_onehot.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
                        feature = vae.sample(mean, logvar).squeeze(0)
                        if not os.path.exists(os.path.join(self.root_path, "vae_feature")):
                            os.mkdir(os.path.join(self.root_path, "vae_feature"))
                        torch.save(feature, os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index))
                        self._vae_feature = feature
                else:
                    logging.error("Vae Feature %s does not exist" % os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index))
                    raise FileNotFoundError("Vae Feature %s does not exist" % os.path.join(self.root_path, "vae_feature", "%04d.pt" % self.index))
        return self._vae_feature
    
    def transform_waypoints(self,x,y,x_command,y_command,theta):
        x_relative = x_command - x
        y_relative = y_command - y
        x = x_relative * np.cos(theta) + y_relative * np.sin(theta)
        y = -x_relative * np.sin(theta) + y_relative * np.cos(theta)
        return (x,y)       

    @property
    def command_waypoints(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        x = self._measurements["gps_x"]
        y = self._measurements["gps_y"]
        theta = self._measurements["theta"]
        waypoints = []
        for i in self._measurements["future_waypoints"][0:self.pred_len]:
            way_x,way_y = self.transform_waypoints(x,y,i[0],i[1],theta)
            waypoints.append((way_x,way_y))
        return waypoints
    
    def GetGPSPoint(self,idx:int):
        if self.cache is not None:
            assert isinstance(self.cache,dict)
            key = "/".join(self.root_path.replace("\\","/").split("/")[-3:])
            if key in self.cache.keys():
                x,y,theta = self.cache[key]['po'][idx]
                return x,y,theta
        measurements = self._LoadJson("measurements_full", idx)
        x = measurements["gps_x"]
        y = measurements["gps_y"]
        theta = measurements["theta"]
        return x,y,theta
            
    @property
    def future_waypoints(self):
        x,y,theta = self.GetGPSPoint(self.index)
        waypoints = []
        for i in range(self.pred_len):
            future_x,future_y,_ = self.GetGPSPoint(self.index+i+1)
            way_x,way_y = self.transform_waypoints(x,y,future_x,future_y,theta)
            waypoints.append((way_x,way_y))
        return torch.Tensor(waypoints)
    
    @property
    def command_waypoints(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        
        x = self._measurements["gps_x"]
        y = self._measurements["gps_y"]
        theta = self._measurements["theta"]
        command_waypoints = []
        for i in range(min(self.pred_len, len(self._measurements["future_waypoints"]))):
            waypoint = self._measurements["future_waypoints"][i]
            way_x,way_y = self.transform_waypoints(x,y,waypoint[0],waypoint[1],theta)
            command_waypoints.append((way_x,way_y))
        for i in range(self.pred_len-len(self._measurements["future_waypoints"])):
            command_waypoints.append((0,0))
        return torch.Tensor(command_waypoints)

    @property
    def should_break(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["should_brake"] == 1:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def should_slow(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["should_slow"] == 1:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def is_junction(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_junction"] == True:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
        
    @property
    def is_vehicle_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_vehicle_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def is_bike_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_bike_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def is_lane_vehicle_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_lane_vehicle_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
        
    @property
    def is_junction_vehicle_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_junction_vehicle_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
        
    @property
    def is_pedestrian_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_pedestrian_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def is_red_light_present(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.index)
        if self._measurements["is_red_light_present"] != []:
            return torch.Tensor([1])
        else:
            return torch.Tensor([0])
    
    @property
    def stop_reason_onehot(self):
        return torch.cat([self.should_break,self.should_slow,self.is_junction,self.is_vehicle_present,self.is_bike_present,self.is_lane_vehicle_present,self.is_junction_vehicle_present,self.is_pedestrian_present,self.is_red_light_present],dim=0).to(torch.float32)
    
if __name__ == "__main__":
    # a = CarlaLabel("test/data/weather-0/data/routes_town01_long_w0_06_23_01_05_07", 0,vae_model_path='pretrained/vae_one_hot/vae_model_54.pth')
    a = CarlaLabel("E:\\remote\\dataset-full\\weather-0\\data\\routes_town01_long_w0_06_23_00_31_21", 65,pred_len=4)
    print(a.future_waypoints)
    # print(a.command_waypoints)
    # print(a.measurements_onehot)
    # print(a.measurements_onehot.shape)