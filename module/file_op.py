import os
import random
import glob
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
import pyodi.apps.coco as pyodi
import json
def CreateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Path created: {dir}')
        
        
def randomPickImage(coco, amount, src_images):
    list = []
    train_list = glob.glob(f'{src_images}/*.[jp][pn][g]')

    for i, image in enumerate(train_list):
        image = image.split('/')[-1]
        train_list[i] = image
        
    images_list = random.choices(train_list, k = amount)
    
    for image in images_list:
        for i, j in coco.imgs.items():
            if image == j['file_name']:
                list.append(j['id'])

    return list


def copy_image(src, des):
    files = glob.glob(f'{src}/*.[jp][pn][g]')
    for src in files:
        file = src.split('/')[-1]
        shutil.copyfile(src, f'{des}/{file}')


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    
    
def cleanDst(dst_images):
    if os.path.exists(dst_images):
        shutil.rmtree(dst_images)

    
    
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
    
    