import torch,sys,cv2,time
import numpy as np
from PIL import Image
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./dataset')
from models.vae import VAE
from models.unet import UNet
from dataset.carla_dataset import CarlaDataset
from torchvision.transforms import ToTensor,Resize,InterpolationMode,CenterCrop,Normalize,Compose
import clip
from diffusers import PNDMScheduler
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
vae_param = torch.load('pretrained/vae_one_hot/vae_model_54.pth')['model_state_dict']
vae_model.load_state_dict(vae_param)
vae_model.eval()

UNet_model = UNet().to(device)
UNet_param = torch.load('pretrained/diffusion/diffusion_model_8.pth')['model_state_dict']
UNet_model.load_state_dict(UNet_param)
UNet_model.eval()
scheduler = PNDMScheduler(
            num_train_timesteps=1000, 
            beta_end=0.012, 
            beta_start=0.00085,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            set_alpha_to_one=False,
            skip_prk_steps=True,
            steps_offset=1,
            trained_betas=None
            )

clip_encoder, _ = clip.load("ViT-L/14", device=device)
clip_encoder.eval()
preprocess = Compose([
                Resize(224, interpolation=InterpolationMode.BILINEAR),
                CenterCrop(224),
                Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                ])
ds = CarlaDataset('test/data',weathers=[0],towns=[1],topdown_base_weight=1,topdown_diff_weight=100)

for (data,label) in ds:
    with torch.no_grad():
        data.image_front,data.image_left,data.image_right,data.image_far,label.topdown
        image_front = data._image_front
        image_left = data._image_left
        image_right = data._image_right
        image_far = data._image_far
        width, height = image_front.size
        result = Image.new('RGB', (width*3, height))
        result.paste(image_left, (0, 0, width, height))
        result.paste(image_front, (width, 0, width*2, height))
        result.paste(image_right, (width*2, 0, width*3, height))
        result = np.array(result)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        topdown_gt = np.array(label._topdown.convert('RGB'))
        topdown_gt = cvt_rgb_seg(topdown_gt)
        
        image_front = ToTensor()(image_front).unsqueeze(0).to(device)
        image_left = ToTensor()(image_left).unsqueeze(0).to(device)
        image_right = ToTensor()(image_right).unsqueeze(0).to(device)
        image_far = ToTensor()(image_far).unsqueeze(0).to(device)
        image_full_tensor = torch.cat(
        (preprocess(image_front), 
                preprocess(image_left), 
                preprocess(image_right), 
                preprocess(image_far)), dim=0)
        pos_clip_feature = clip_encoder.encode_image(image_full_tensor).unsqueeze(0)
        # neg_clip_feature = clip_encoder.encode_image(torch.zeros_like(image_full_tensor)).unsqueeze(0)
        # out_clip_feature = torch.cat((neg_clip_feature,pos_clip_feature),dim=0).to(torch.float32)
        out_clip_feature = pos_clip_feature.to(torch.float32)
        out_vae = torch.randn(1,4,32,32).to(device)
        scheduler.set_timesteps(100, device=device)
        start_time = time.time()
        for cur_time in scheduler.timesteps:
            # noise = torch.cat((out_vae,out_vae),dim=0)
            noise = out_vae
            noise = scheduler.scale_model_input(noise, cur_time)
            pred_noise = UNet_model(out_vae=noise,out_encoder=out_clip_feature,time=cur_time)
            # pred_noise = pred_noise[1]
            # pred_noise = pred_noise[0] + 2 * (pred_noise[1] - pred_noise[0])
            out_vae = scheduler.step(pred_noise, cur_time,out_vae).prev_sample
        print("Time: %.4f"%(time.time()-start_time))
        # out_vae = 1 / 0.18215 * out_vae
        out_vae = vae_model.decoder(out_vae).squeeze(0)
        out_vae = torch.nn.functional.softmax(out_vae,dim=0)
        out_vae = torch.argmax(out_vae,dim=0).unsqueeze(0)
        out_vae = (torch.cat((out_vae,out_vae,out_vae))).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        out_vae = cvt_rgb_seg(out_vae)
        
    cv2.imshow('topdown',out_vae)
    cv2.imshow('topdown_gt',topdown_gt)
    cv2.imshow('result',result)
    cv2.waitKey(1)
