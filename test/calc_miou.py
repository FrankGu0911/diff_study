import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Value
from tqdm.contrib.concurrent import process_map
import logging
seg_tag = {
    0: [0,0,0], # Unlabeled
    1: [70,70,70], # Building
    2: [100,40,40], # Fence
    3: [55,90,80], # Other
    4: [220,120,60], # Pedestrian
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

drivable_area = [7,6,15,16]
pedestrian = [4]
car = [10]
sidewalk = [8]
intersection=Value('i',0)
union=Value('i',0)
RESULT_PATH = "test/infer/lidar_100"
gt_path = os.path.join(RESULT_PATH, "gt")
result_path = os.path.join(RESULT_PATH, "result")
def judge_type(rgb:np.ndarray):
    rgb = rgb.astype(np.int32)
    for i in seg_tag:
        if np.array_equal(rgb, np.array(seg_tag[i])):
            return i
    if np.array_equal(rgb,np.array([26,26,26])):
        return 7
    raise ValueError("rgb is not in seg_tag")

def calculate_single_iou(index,calc_area):
    gt = cv2.imread(os.path.join(gt_path, "%04d.png" % index))
    results = cv2.imread(os.path.join(result_path, "%04d.png" % index))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    results = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
    gt = gt.reshape(-1, 3)
    results = results.reshape(-1, 3)
    gt = np.apply_along_axis(judge_type, 1, gt)
    results = np.apply_along_axis(judge_type, 1, results)
    global intersection
    global union
    for index,result in enumerate(results):
        # calculate drivable iou
        cur_gt = gt[index]
        if (result in calc_area) and (cur_gt in calc_area):
            intersection.value += 1
            union.value += 1
        elif (result in calc_area) or (cur_gt in calc_area):
            union.value += 1
        else:
            pass

logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    num = len(os.listdir(gt_path))
    process_map(calculate_single_iou,[i for i in range(num)],[drivable_area]*num,max_workers=32,chunksize=64)
    print("drivable_area: %.4f, %d/%d" % (intersection.value/union.value,intersection.value,union.value))
    intersection.value = 0
    union.value = 0
    process_map(calculate_single_iou,[i for i in range(num)],[pedestrian]*num,max_workers=32,chunksize=64)
    print("pedestrian: %.4f, %d/%d" % (intersection.value/union.value,intersection.value,union.value))
    intersection.value = 0
    union.value = 0
    process_map(calculate_single_iou,[i for i in range(num)],[car]*num,max_workers=32,chunksize=64)
    print("car: %.4f, %d/%d" % (intersection.value/union.value,intersection.value,union.value))
    intersection.value = 0
    union.value = 0
    process_map(calculate_single_iou,[i for i in range(num)],[sidewalk]*num,max_workers=32,chunksize=64)
    print("sidewalk: %.4f, %d/%d" % (intersection.value/union.value,intersection.value,union.value))
    intersection.value = 0
    union.value = 0
    