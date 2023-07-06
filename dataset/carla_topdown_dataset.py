from base_io_dataset import BaseIODataset
import os,re,logging,torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
_logger = logging.getLogger(__name__)
class CarlaTopDownDataset(BaseIODataset):
    def __init__(
        self,
        root,
        img_size=(256, 256),
        weathers=[0,1, 3, 6, 8],
        towns=[1,2,3]
        ):
        super().__init__(root)
        self.img_size = img_size
        dataset_indexs = self._load_text(os.path.join(root, 'dataset_index.txt')).split('\n')
        pattern = re.compile('weather-(\d+).*town(\d\d)')
        self.route_frames = []
        for line in dataset_indexs:
            if len(line.split()) != 2:
                continue
            path, frames = line.split()
            frames = int(frames)
            res = pattern.findall(path)
            if len(res) != 1:
                continue
            weather = int(res[0][0])
            town = int(res[0][1])
            if weather not in weathers or town not in towns:
                continue
            for i in range(frames):
                self.route_frames.append((os.path.join(root, path), i))
        _logger.info("Sub route dir nums: %d" % len(self.route_frames))
    
    def __len__(self):
        return len(self.route_frames)
    
    def __getitem__(self, idx):
        data = {}
        route_dir, frame_id = self.route_frames[idx]
        
        topdown_img = self._load_image(os.path.join(route_dir, 'topdown', '%04d.png' % frame_id))
        tar_x, tar_y = self.img_size
        topdown_img = topdown_img.crop(self.calc_crop(tar_x, tar_y))
        return transforms.ToTensor()(topdown_img), torch.Tensor([0])
        
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

if __name__ == '__main__':
    ds = CarlaTopDownDataset('test/data')
    dl = DataLoader(ds,batch_size=4,shuffle=True)
    for i in dl:
        print(i.shape)
        break