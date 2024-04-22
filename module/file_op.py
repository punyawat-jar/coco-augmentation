import os
import random
import glob
import shutil
import numpy as np
import cv2
from pathlib import Path

def CreateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Path created: {dir}')
        
        
def randomPickImage(keys, amount):
    list = random.choices(keys, k =amount)
    return list


def copy_image(src, des):
    files = glob.glob(f'{src}/*.[jp][pn][g]')
    for src in files:
        file = src.split('/')[-1]
        shutil.copyfile(src, f'{des}/{file}')


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    
    
def cleanDst(dst_images, src_json, dst_json):
    if os.path.exists(dst_images):
        shutil.rmtree(dst_images)
    if os.path.exists(dst_json):
        os.remove(dst_json)
    
    
def load_image_coco(coco, key, file_type):
    file_name = coco.loadImgs(key)[0]['file_name']
    image = cv2.imread(f'{file_type}/{file_name}', cv2.COLOR_BGR2GRAY)
    return file_name, image
            
def mkdir(dir):
    path = Path(dir)
    path.mkdir(parents = True, exist_ok = True)
    
    
def read_corr_coco(coco, key):
    item = []
    for _, value in coco.anns.items():
        if value['image_id'] == key:
            item.append(value)
            
    return item