import os
import re
import logging
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor,Resize,InterpolationMode,CenterCrop,Normalize,Compose
load_list = {
    "seg_right": "%04d.png",
    "seg_left": "%04d.png",
    "seg_front": "%04d.png",
    "depth_right": "%04d.png",
    "depth_left": "%04d.png",
    "depth_front": "%04d.png",
    "lidar": "%04d.npy",
    "measurements_full": "%04d.json"
}

class CarlaData():
    def __init__(
        self, 
        path: str, 
        idx: int, 
        is_rgb_merged: bool = True,
        gen_feature: bool = False,
        seq_len: int = 1,
        cache = None
        ):
        self.root_path = path
        self.relative_path = "/".join(self.root_path.replace("\\","/").split("/")[-3:])
        self.idx = idx
        self.gen_feature = gen_feature
        self.seq_len = seq_len
        self.cache = cache
        self._image_front = None
        self._image_left = None
        self._image_right = None
        self._image_rear = None
        self._image_far = None
        self._seg_front = None
        self._seg_left = None
        self._seg_right = None
        self._depth_front = None
        self._depth_left = None
        self._depth_right = None
        self._lidar = None
        self._lidar_2d = None
        self._measurements = None
        self._is_rgb_merged = is_rgb_merged
        self._rgb_merged = None
        self._clip_feature = None
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __repr__(self) -> str:
        return "Data: %d at %s" % (self.idx, self.root_path)

    def __str__(self) -> str:
        return "Data: %d at %s" % (self.idx, self.root_path)

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

    def _LoadNpy(self, name: str, idx: int):
        path = os.path.join(self.root_path, name, "%04d.npy" % idx)
        if not os.path.exists(path):
            logging.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")
        logging.debug(f"Load npy from {path}")
        return np.load(path)

    @property
    def image_front(self):
        if self._image_front is None:
            if self._is_rgb_merged:
                if self._rgb_merged is None:
                    self._rgb_merged = self._LoadImage("rgb_full", self.idx)
                logging.debug(f"Crop image_front from rgb_full")
                self._image_front = self._rgb_merged.crop((0, 0, 800, 600))
            else:
                self._image_front = self._LoadImage("rgb_front", self.idx)
        return ToTensor()(self._image_front).unsqueeze(0)
    
    @property
    def image_left(self):
        if self._image_left is None:
            if self._is_rgb_merged:
                if self._rgb_merged is None:
                    self._rgb_merged = self._LoadImage("rgb_full", self.idx)
                logging.debug(f"Crop image_left from rgb_full")
                self._image_left = self._rgb_merged.crop((0, 600, 800, 1200))
            else:
                self._image_left = self._LoadImage("rgb_left", self.idx)
        return ToTensor()(self._image_left).unsqueeze(0)

    @property
    def image_right(self):
        if self._image_right is None:
            if self._is_rgb_merged:
                if self._rgb_merged is None:
                    self._rgb_merged = self._LoadImage("rgb_full", self.idx)
                logging.debug(f"Crop image_right from rgb_full")
                self._image_right = self._rgb_merged.crop((0, 1200, 800, 1800))
            else:
                self._image_right = self._LoadImage("rgb_right", self.idx)
        return ToTensor()(self._image_right).unsqueeze(0)
    
    @property
    def image_far(self):
        if self._image_far is None:
            if self._is_rgb_merged:
                if self._rgb_merged is None:
                    self._rgb_merged = self._LoadImage("rgb_full", self.idx)
                logging.debug(f"Crop image_far from rgb_full")
                self._image_far = self._rgb_merged.crop((200, 150, 600, 450))
            else:
                self._image_far = self._LoadImage("rgb_far", self.idx)
        return Resize((600,800),interpolation=InterpolationMode.NEAREST)(ToTensor()(self._image_far)).unsqueeze(0)

    @property
    def image_rear(self):
        if self._image_rear is None:
            self._image_rear = self._LoadImage("rgb_rear", self.idx)
        return ToTensor()(self._image_rear).unsqueeze(0)
    
    @property
    def image_full(self):
        return torch.cat((self.image_front, self.image_left, self.image_right,self.image_far), dim=0)

    @property
    def seg_front(self):
        if self._seg_front is None:
            self._seg_front = self._LoadImage("seg_front", self.idx, "png")
        return self._seg_front
    
    @property
    def seg_left(self):
        if self._seg_left is None:
            self._seg_left = self._LoadImage("seg_left", self.idx, "png")
        return self._seg_left

    @property
    def seg_right(self):
        if self._seg_right is None:
            self._seg_right = self._LoadImage("seg_right", self.idx, "png")
        return self._seg_right
    
    @property
    def depth_front(self):
        if self._depth_front is None:
            self._depth_front = self._LoadImage("depth_front", self.idx, "png")
        return ToTensor()(self._depth_front)
    
    @property
    def depth_left(self):
        if self._depth_left is None:
            self._depth_left = self._LoadImage("depth_left", self.idx, "png")
        return ToTensor()(self._depth_left)
    
    @property
    def depth_right(self):
        if self._depth_right is None:
            self._depth_right = self._LoadImage("depth_right", self.idx, "png")
        return ToTensor()(self._depth_right)
    
    @property
    def lidar(self):
        if self._lidar is None:
            self._lidar = self._LoadNpy("lidar", self.idx)
        return ToTensor()(self._lidar)
    
    def splat_points(self,point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    def lidar_to_histogram_features(self,lidar, crop=256):
        """
        Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
        """
        below = lidar[lidar[..., 2] <= -2.0]
        above = lidar[lidar[..., 2] > -2.0]
        below_features = self.splat_points(below)
        above_features = self.splat_points(above)
        total_features = below_features + above_features
        features = np.stack([below_features, above_features, total_features], axis=-1).astype(np.float32)
        # features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features

    def lidar_2d_feature(self, lidar:np.ndarray):
        point = lidar[:,:3]
        point = point[(point[:,2] > -2.0)] # above road
        point = point[(point[:,0] > -22.5)]
        point = point[(point[:,0] < 22.5)]
        point = point[(point[:,1] < 45)]
        point = point[(point[:,1] > 0)]
        histogram = np.zeros((256,256))
        for p in point:
            histogram[int((45-p[1])/45*256),int((p[0]+22.5)/45*256)] = 255
        return histogram

    @property
    def lidar_2d(self):
        if self._lidar_2d is None:
            try:
                self._lidar_2d = self._LoadImage("lidar_2d", self.idx, "png")
            except FileNotFoundError:
                if self._lidar is None:
                    self._lidar = self._LoadNpy("lidar", self.idx)
                histogram = self.lidar_2d_feature(self._lidar)
                self._lidar_2d = Image.fromarray(histogram).convert('L')
                if not os.path.exists(os.path.join(self.root_path, "lidar_2d")):
                    os.makedirs(os.path.join(self.root_path, "lidar_2d"))
                self._lidar_2d.save(os.path.join(self.root_path, "lidar_2d", "%04d.png" % self.idx))
        ret = ToTensor()(self._lidar_2d)
        if ret.shape[0] == 1:
            ret = torch.cat((ret, ret, ret))
        return ret

    @property
    def clip_feature(self):
        if self._clip_feature is None:
            if os.path.exists(os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx)):
                self._clip_feature = torch.load(os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx),map_location='cpu').to(torch.float32)
            else:
                logging.debug(f"Clip feature file {os.path.join(self.root_path, 'clip_feature', '%04d.pt' % self.idx)} does not exist")
                if not os.path.exists(os.path.join(self.root_path, "clip_feature")):
                    os.makedirs(os.path.join(self.root_path, "clip_feature"))
                if self.gen_feature:
                    import clip
                    clip_encoder,_ = clip.load("ViT-L/14", device='cuda' if torch.cuda.is_available() else 'cpu')
                    preprocess = Compose([
                        Resize(224, interpolation=InterpolationMode.BILINEAR),
                        CenterCrop(224),
                        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                    ])
                    clip_encoder.eval()
                    with torch.no_grad():
                        self._clip_feature = clip_encoder.encode_image(preprocess(self.image_full.to('cuda' if torch.cuda.is_available() else 'cpu')))
                    torch.save(self._clip_feature, os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx))
        return self._clip_feature

    @property
    def ego_position(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        (x,y,theta) = self._measurements["gps_x"],self._measurements["gps_y"],self._measurements["theta"]
        if np.isnan(theta):
            print(self.root_path,self.idx)
            with open("log/data.log","a") as f:
                f.write("%s %d\n" % (self.root_path,self.idx))
            # raise ValueError("theta is nan")
            if self.idx != 0:
                me = self._LoadJson("measurements_full", self.idx-1)
                theta1 = me["theta"]
            if self.idx != len(os.listdir(os.path.join(self.root_path, "measurements_full"))) - 1:
                me = self._LoadJson("measurements_full", self.idx + 1)
                theta2 = me["theta"]
            if self.idx == 0:
                raise ValueError("theta is nan")
            elif self.idx == len(os.listdir(os.path.join(self.root_path, "measurements_full"))) - 1:
                theta = theta1
            elif np.isnan(theta2):
                theta = theta1
            else:
                # calucate the acute angle between theta1 and theta2
                import math
                if math.fabs(theta1-theta2) > math.pi:
                    theta = (theta1 + theta2) / 2 + math.pi
                else:
                    theta = (theta1 + theta2) / 2
            self._measurements["theta"] = theta
            json.dump(self._measurements,open(os.path.join(self.root_path, "measurements_full", "%04d.json" % self.idx),'w'))
    
        return (x,y,theta)
    
    @property
    def ego_x(self):
        return self.ego_position[0]

    @property
    def ego_y(self):
        return self.ego_position[1]
    
    @property
    def ego_theta(self):
        return self.ego_position[2]

    @property
    def raw_point_command(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        return (self._measurements["x_command"],self._measurements["y_command"])
    
    @property
    def raw_x_command(self):
        return self.raw_point_command[0]
    
    @property
    def raw_y_command(self):
        return self.raw_point_command[1]
    
    @property
    def point_command(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        x_relative = self.raw_x_command - self.ego_x
        y_relative = self.raw_y_command - self.ego_y
        x = x_relative * np.cos(self.ego_theta) + y_relative * np.sin(self.ego_theta)
        y = -x_relative * np.sin(self.ego_theta) + y_relative * np.cos(self.ego_theta)
        ret = torch.tensor((x,y))
        if torch.isnan(ret).any():
            print(self.root_path,self.idx)
            print(self.ego_position)
            print(self.raw_point_command)
            raise ValueError("point_command is nan")
        return ret
    
    @property
    def x_command(self):
        return self.point_command[0]
    
    @property
    def y_command(self):
        return self.point_command[1]

    @property
    def speed(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        return self._measurements["speed"]
    
    @property
    def command(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        return self._measurements["command"]

    @property
    def gt_command(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        return self._measurements["gt_command"]
    
    @property
    def gt_command_onehot(self):
        if self._measurements is None:
            self._measurements = self._LoadJson("measurements_full", self.idx)
        # 1 - 6
        ret = torch.zeros(6,dtype=torch.float32)
        ret[self._measurements["gt_command"] - 1] = 1
        if torch.isnan(ret).any():
            raise ValueError("gt_command_onehot is nan")
        return ret
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    data = CarlaData("E:\\remote\\dataset-full\\weather-0\\data\\routes_town01_long_w0_06_23_00_31_21", 0)
    print(data.gt_command_onehot)