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
    # print(seg.shape)
    for i in seg_tag:
      seg = np.where(seg == [i,i,i], np.array(seg_tag[i]), seg)

    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg

vae_model = VAE(26,26).to(device)
model = torch.load('pretrained/vae_one_hot/vae_model_10.pth')['model_state_dict']
vae_model.load_state_dict(model)

ds = CarlaTopDownDataset('test/data',onehot=True,weathers=[0],base_weight=1,diff_weight=100)
ds_no_onehot = CarlaTopDownDataset('test/data',onehot=False,weathers=[0])

for i,(image,_) in enumerate(ds):
    image_np = ds_no_onehot[i][0]
    image_np = (torch.cat((image_np,image_np,image_np))*torch.tensor(25)).permute(1, 2, 0).numpy().astype(np.uint8)

    image_neo = vae_model(image.unsqueeze(0).to(device))[0].squeeze(0)
    #softmax
    image_neo = torch.nn.functional.softmax(image_neo,dim=0)
    image_neo = torch.argmax(image_neo,dim=0).unsqueeze(0)
    image_neo = (torch.cat((image_neo,image_neo,image_neo))).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # print(image_neo.shape)
    
    # image_neo = torch.cat((image_neo,image_neo,image_neo),dim=1).squeeze(0)
    # image_neo = image_neo.permute(1, 2, 0)
    # image_neo_show = (image_neo.detach()*25).cpu().clone().numpy().astype(np.uint8)
    image_np = cvt_rgb_seg(image_np)
    image_neo = cvt_rgb_seg(image_neo)
    cv2.imshow('infer',image_neo)
    cv2.imshow('groundtruth',image_np)
    cv2.waitKey(33)
    
    
    