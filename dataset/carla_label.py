import os,re,logging,torch
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
        ):
        self.root_path = path
        self.index = index
        self._topdown = None
        self._topdown_onehot = None
        self.important_seg = important_seg
        self.base_weight = base_weight
        self.diff_weight = diff_weight


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
            self._topdown = (transforms.ToTensor()(self._topdown) * 255).to(torch.uint8)
            if torch.max(self.topdown) > 25:
                _logger.debug("Topdown image has value larger than 25: %s %s" % (self.root_path, self.index))
                # replace with 7
                self._topdown = torch.where(self._topdown > 25, torch.Tensor([7]).to(torch.uint8), self._topdown)

        return self._topdown
    
    @property
    def topdown_onehot(self):
        if self._topdown_onehot is None:
            self._topdown_onehot = self.get_one_hot(self.topdown, 26).to(torch.float32)
        return self._topdown_onehot
    
if __name__ == "__main__":
    a = CarlaLabel("test/data/dataset/weather-0/data/routes_town01_long_w0_06_23_00_31_21", 0)
    print(a.topdown_onehot.shape)
    print(a.topdown.shape)