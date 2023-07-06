import torch,sys,cv2
import numpy as np
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.vae import VAE
from dataset.carla_topdown_dataset import CarlaTopDownDataset
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_tag = {
    0: [0,0,0], # Unlabeled
    1: [70,70,70], # Building
    2: [100,40,40], # Fence
    3: [55,90,80], # Other
    4: [220,20,60], # Pedestrian
    5: [153,153,153], # Pole
    6: [157,234,50], # RoadLine
    7: [128,64,128], # Road
    8: [244,35,232], # Sidewalk
    9: [107,142,35], # Vegetation
    10: [0,0,142], # Car
    11: [102,102,156], # Wall
    12: [220,220,0], # TrafficSign
    13: [70,130,180], # Sky
    14: [81,0,81],  # Ground
    15: [150,100,100], # Bridge
    16: [230,150,140], # RailTrack
    17: [180,165,180], # GuardRail
    18: [250,170,30], # TrafficLight
    19: [110,190,160], # Static
    20: [170,120,50], # Dynamic
    21: [45,60,150], # Water
    22: [145,170,100], # Terrain
    23: [255,0,0], # RedLight
    24: [255,255,0], # YellowLight
    25: [0,255,0], # GreenLight
}

def cvt_rgb_seg(seg:np.ndarray):
    print(seg.shape)
    for i in seg_tag:
      seg = np.where(seg == i, np.array([i,i,i]), seg)

    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg

vae_model = VAE(1,1).to(device)
vae_model.load_state_dict(torch.load('vae_model_7.pth'))

ds = CarlaTopDownDataset('test/data')
# print((ds[0][0]*256).char())
image = ds[0][0]
image_neo = vae_model(image.unsqueeze(0).to(device))
image_neo_show = (image_neo[0].detach()*256).cpu().clone().squeeze(0).numpy().astype(np.uint8)
# image_neo_show = cv2.cvtColor(image_neo_show, cv2.COLOR_GRAY2RGB)
# cv2.imshow(image_neo_show)
# print(image_neo_show)
# for image,_ in ds:
#     # seg_img = cvt_rgb_seg(image.numpy().squeeze(0))
#     # cv2.imshow('seg_bev_crop',seg_img)
#     # cv2.waitKey(33)
#     image_neo = vae_model(image.unsqueeze(0).to(device))
#     image_neo_show = image_neo.cpu().clone().numpy()
    
    