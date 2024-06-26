import os

import albumentations as A

import shutil

from pathlib import Path

from tqdm import tqdm
import traceback

from module.file_op import *
from module.display import *
from module.mask import *
from module.augment import *


class Augmentation:
    def __init__(self, src = None, dst = None, src_json = None):
        self.src = src
        self.dst = dst
        self.src_json = src_json
        
        self.dst_json = None
        self.dst_images = None

        self.amount = 1

    def setFolder(self, src, dst):
        self.src = src
        self.dst = dst

    def setSrcjson(self, src_json):
        self.src_json = src_json
    
    def setAugLib(self, auglib):
        self.auglib = auglib
        
    def run(self, amount, augmentation):
        try:
            self.dst_images = f'{self.dst}/images'
            self.dst_json = f'{self.dst}/data.json'
            self.amount = amount
            
            if not os.path.exists(self.src_json):
                raise Exception(f"Source file not found: {self.src_json}")
            
            cleanDst(self.dst, self.src_json, self.dst_json)
            
            mkdir(self.dst)
            mkdir(self.dst_images)
            shutil.copyfile(self.src_json, self.dst_json)
            
            createAugmentation(augmentation, self.src, self.dst_images, self.dst_json, amount)
            copy_image(self.src, self.dst_images)
        except Exception as e:
            print(e)
            traceback.print_exc()
def main():


    
    augmentation = A.Compose([
        A.RandomCrop(width=640, height=640),
        # A.HorizontalFlip(p=0.5),
        # A.CLAHE(clip_limit=4.0, tile_grid_size=(16,16), always_apply = True, p = 1.0)
    ])
    amount = 2
    
    
    src = f'./source/AP/images'
    dst = f'./augmentation/AP'
    src_json = f'./source/AP/AP.json'
    
    
    
    augment = Augmentation()
    augment.setFolder(src, dst)
    augment.setSrcjson(src_json)
    
    augment.run(amount, augmentation)
    

if __name__ == '__main__':
    main()

