import os,re,logging,torch,json
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
            topdown_base_weight=1,
            topdown_diff_weight=100,
            gen_feature=False,
            vae_model_path=None,
            pred_len=0,
            seq_len=1,
            use_cache=True,
    ):
        super().__init__()
        self.root = root
        self.dataset_indexs = open(os.path.join(root, 'dataset_index.txt'), 'r').read().split('\n')
        pattern = re.compile('weather-(\d+).*town(\d\d)')
        self.route_frames = []
        self.topdown_onehot = topdown_onehot
        self.topdown_base_weight = topdown_base_weight
        self.topdown_diff_weight = topdown_diff_weight
        self.gen_feature = gen_feature
        self.vae_model_path = vae_model_path
        self.pred_len = pred_len
        self.seq_len = seq_len
        if use_cache:
            if not os.path.exists(os.path.join(root,'cache.json')):\
                self.gen_cache()
            self.cache = json.load(open(os.path.join(root,'cache.json'),'r'))
        else:
            self.cache = None
        for line in self.dataset_indexs:
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
            # remove the first frame and the last two frame
            for i in range(self.seq_len + 1, frames - self.pred_len - 1):
                self.route_frames.append((os.path.join(root, path), i))
        _logger.info("Sub route dir nums: %d" % len(self.route_frames))

    def __len__(self):
        return len(self.route_frames)
    
    def __getitem__(self, idx):
        route_dir, frame_id = self.route_frames[idx]
        data = CarlaData(route_dir, frame_id, 
                         gen_feature=self.gen_feature,
                         seq_len=self.seq_len,
                         cache = self.cache)
        label = CarlaLabel(route_dir, frame_id,
                           base_weight=self.topdown_base_weight,
                           diff_weight=self.topdown_diff_weight, 
                           gen_feature=self.gen_feature, 
                           vae_model_path=self.vae_model_path,
                           pred_len=self.pred_len,
                           cache=self.cache)
        return (data, label)

    def gen_cache(self):
        self.cache = {}
        from tqdm import tqdm
        for line in tqdm(self.dataset_indexs):
            if len(line.split()) != 2:
                continue
            path, frames = line.split()
            frames = int(frames)
            if not os.path.exists(os.path.join(self.root,path)):
                logging.warning("Path %s not exists" % os.path.join(self.root,path,'measurements_full', "%04d.json" % frames))
                continue
            points = []
            for i in range(frames):
                route_dir = os.path.join(self.root,path)
                data = CarlaData(route_dir, i, 
                                 gen_feature=self.gen_feature,
                                 seq_len=self.seq_len,
                                 cache = self.cache)
                points.append(data.ego_position)
            self.cache[path] = points
        json.dump(self.cache,open(os.path.join(self.root,'cache.json'),'w'))

    @staticmethod
    def image2topdown_collate_fn(batch):
        data = torch.cat([data.image_full.unsqueeze(0)
                         for (data, label) in batch], dim=0)
        label = torch.cat([label.topdown_onehot.unsqueeze(0)
                          for (data, label) in batch], dim=0)
        return (data, label)
    
    @staticmethod
    def clip_feature2vae_feature_collate_fn(batch):
        try:
            data = torch.cat([data.clip_feature.unsqueeze(0)
                         for (data, label) in batch], dim=0)
            label = torch.cat([label.vae_feature.unsqueeze(0)
                              for (data, label) in batch], dim=0)
        except:
            for (data, label) in batch:
                print('clip_feature:',data.clip_feature.shape)
                print('vae_feature:',label.vae_feature.shape)
        return (data, label)
    
    @staticmethod
    def clip_lidar_feature2vae_feature_collate_fn(batch):
        try:
            data = []
            data.append(torch.cat([data.clip_feature.unsqueeze(0)
                         for (data, label) in batch], dim=0))
            data.append(torch.cat([data.lidar_2d_feature.unsqueeze(0)
                         for (data, label) in batch], dim=0))
            label = torch.cat([label.vae_feature.unsqueeze(0)
                              for (data, label) in batch], dim=0)
        except Exception as e:
            for (data, label) in batch:
                print("data_path: %s:%d" %(data.root_path,data.idx))
                print('clip_feature:',data.clip_feature.shape)
                print('vae_feature:',label.vae_feature.shape)
            raise e
        return (data, label)

    @staticmethod
    def clip_lidar2d_feature2vae_feature_collate_fn(batch):
        try:
            data = []
            data.append(torch.cat([data.clip_feature.unsqueeze(0)
                         for (data, label) in batch], dim=0))
            data.append(torch.cat([data.lidar_2d.unsqueeze(0)
                         for (data, label) in batch], dim=0))
            label = torch.cat([label.vae_feature.unsqueeze(0)
                              for (data, label) in batch], dim=0)
        except Exception as e:
            for (data, label) in batch:
                print("data_path: %s:%d" %(data.root_path,data.idx))
                print('clip_feature:',data.clip_feature.shape)
                print('vae_feature:',label.vae_feature.shape)
            raise e
        return (data, label)
    
    @staticmethod
    def vae_measurement2wp_collate_fn(batch):
        try:
            data = []
            data.append(torch.cat([label.vae_feature.unsqueeze(0)
                         for (data, label) in batch], dim=0))
            data.append(torch.cat([torch.cat([data.point_command,data.gt_command_onehot]).unsqueeze(0)
                         for (data, label) in batch], dim=0))
            label = torch.cat([label.future_waypoints.unsqueeze(0)
                              for (data, label) in batch], dim=0)
        except Exception as e:
            for (data, label) in batch:
                print("data_path: %s:%d" %(data.root_path,data.idx))
                print('vae_feature:',label.vae_feature.shape)
                print('measurement:',torch.cat([data.point_command,data.gt_command_onehot]).shape)
                print('waypoint:',label.future_waypoints.shape)
            raise e
        return (data, label)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    # dataset = CarlaDataset("E:\\dataset")
    # print(len(dataset))
    # val_ds = CarlaDataset('E:/dataset',weathers=[0],towns=[10],topdown_base_weight=1,topdown_diff_weight=100)
    # val_loader = DataLoader(val_ds,
    #                         batch_size=4,
    #                         shuffle=True,
    #                         collate_fn=CarlaDataset.clip_lidar_feature2vae_feature_collate_fn,
    #                         )
    # for (data,label) in val_loader:
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(label.shape)
    dataset = CarlaDataset("E:\\remote\\dataset-full",weathers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],towns=[1,2,3,4,5,6,7,10],pred_len=4)
    from tqdm import tqdm
    for i in tqdm(range(0,len(dataset))):
        dataset[i][0].point_command
    # dataset.gen_cache()
    # for data,label in dataset:
    #     # if data.gt_command != data.command:
    #     #     print(data.root_path,data.idx)
    #     if data.command < 1 or data.command > 6:
    #         print(data.root_path,data.idx)
    #     if data.gt_command < 1 or data.gt_command > 6:
    #         print(data.root_path,data.idx)
    # dataloader = DataLoader(dataset, batch_size=8,shuffle=True, collate_fn=CarlaDataset.vae_measurement2wp_collate_fn)
    # for (data, label) in dataloader:
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(label.shape)
    #     break
    # from tqdm import tqdm
    # for i in tqdm(dataset):
    #     i[0].clip_feature
    #     i[1].vae_feature
