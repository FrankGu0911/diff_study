import os
import re
import logging
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
        self, path: str, 
        idx: int, 
        with_lidar: bool = False, 
        with_seg: bool = False, 
        with_depth: bool = False,
        with_rear: bool = False,
        rgb_merged: bool = True
        ):
        self.root_path = path
        self.idx = idx
        self.with_lidar = with_lidar
        self.with_seg = with_seg
        self.with_depth = with_depth
        self.with_rear = with_rear
        self.rgb_merged = rgb_merged
        if not self._CheckData():
            raise ValueError("Data not found")

    def _CheckData(self):
        return self._CheckRGBData() and self._CheckOtherData()

    def _CheckOtherData(self):
        for key in load_list:
            if (not self.with_lidar) and "lidar" in key:
                logging.debug(f"Skip lidar data at {key}")
                continue
            if (not self.with_seg) and "seg" in key:
                logging.debug(f"Skip seg data at {key}")
                continue
            if (not self.with_depth) and "depth" in key:
                logging.debug(f"Skip depth data at {key}")
                continue
            if (not self.with_rear) and "rear" in key:
                logging.debug(f"Skip rear data at {key}")
                continue
            path = os.path.join(self.root_path, key, load_list[key] % self.idx)
            if not os.path.exists(path):
                logging.error("Data %s not found" % path)
                return False
        return True
    
    def _CheckRGBData(self):
        check_dict = {}
        if self.rgb_merged:
            check_dict["rgb_full"] = "%04d.jpg" % self.idx
        else:
            check_dict["rgb_right"] = "%04d.jpg" % self.idx
            check_dict["rgb_left"] = "%04d.jpg" % self.idx
            check_dict["rgb_front"] = "%04d.jpg" % self.idx
        if self.with_rear:
            check_dict["rgb_rear"] = "%04d.jpg" % self.idx
        for key in check_dict:
            path = os.path.join(self.root_path, key, check_dict[key])
            if not os.path.exists(path):
                logging.error("Data %s not found" % path)
                return False
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    data = CarlaData("C:\\dataset\\weather-0\\data\\routes_town01_long_w0_06_23_00_31_21", 45)
    # print(data._CheckData())