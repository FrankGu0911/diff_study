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
        ):
        self.root_path = path
        self.idx = idx
        self.gen_feature = gen_feature
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        return self._lidar
    
    @property
    def lidar_2d(self):
        #TODO: lidar_to_histogram_features
        pass

    @property
    def clip_feature(self):
        if self._clip_feature is None:
            if os.path.exists(os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx)):
                self._clip_feature = torch.load(os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx)).to(torch.float32)
            else:
                logging.debug(f"Clip feature file {os.path.join(self.root_path, 'clip_feature', '%04d.pt' % self.idx)} does not exist")
                if not os.path.exists(os.path.join(self.root_path, "clip_feature")):
                    os.makedirs(os.path.join(self.root_path, "clip_feature"))
                if self.gen_feature:
                    import clip
                    clip_encoder,_ = clip.load("ViT-L/14", device=self.device)
                    preprocess = Compose([
                        Resize(224, interpolation=InterpolationMode.BILINEAR),
                        CenterCrop(224),
                        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                    ])
                    clip_encoder.eval()
                    with torch.no_grad():
                        self._clip_feature = clip_encoder.encode_image(preprocess(self.image_full.to(self.device)))
                    torch.save(self._clip_feature, os.path.join(self.root_path, "clip_feature", "%04d.pt" % self.idx))
        return self._clip_feature

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    data = CarlaData("E:\\dataset\\weather-0\\data\\routes_town01_long_w0_06_23_00_31_21", 45)
    # data = CarlaData("test/data/weather-0/data/routes_town01_long_w0_06_23_01_05_07", 45)
    # print(data.image_full)
    # preprocess = Compose([
    #         Resize(224, interpolation=InterpolationMode.BILINEAR),
    #         CenterCrop(224),
    #         Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    #     ])
    # print(preprocess(data.image_full).shape)
    print(data.clip_feature.shape)