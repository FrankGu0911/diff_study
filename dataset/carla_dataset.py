import os,re,logging,torch
from torch.utils.data import Dataset,DataLoader
from carla_data import CarlaData
from carla_label import CarlaLabel
_logger = logging.getLogger(__name__)

class CarlaDataset(Dataset):
    def __init__(
            self,
            root,
            weathers=[i for i in range(21)],
            towns=[1,2,3,4,5,6,7,10],
            topdown_onehot=True,
            topdown_base_weight = 1,
            topdown_diff_weight = 100,
            ):
        super().__init__()
        self.root = root
        dataset_indexs = open(os.path.join(root, 'dataset_index.txt'), 'r').read().split('\n')
        pattern = re.compile('weather-(\d+).*town(\d\d)')
        self.route_frames = []
        self.topdown_onehot = topdown_onehot
        self.topdown_base_weight = topdown_base_weight
        self.topdown_diff_weight = topdown_diff_weight
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
        route_dir, frame_id = self.route_frames[idx]
        data = CarlaData(route_dir, frame_id)
        label = CarlaLabel(route_dir, frame_id)
        return (data, label)
    
    @staticmethod
    def image2topdown_collate_fn(batch):
        data = torch.cat([data.image_full.unsqueeze(0) for (data, label) in batch], dim=0)
        label = torch.cat([label.topdown_onehot.unsqueeze(0) for (data, label) in batch], dim=0)
        return (data, label)


if __name__ == '__main__':
    dataset = CarlaDataset("E:\\dataset")
    dataloader = DataLoader(dataset, batch_size=8,shuffle=True, collate_fn=CarlaDataset.image2topdown_collate_fn)
    for (data, label) in dataloader:
        print(data.shape)
        print(label.shape)
        break