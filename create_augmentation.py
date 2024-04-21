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

def mkdir(dir):
    path = Path(dir)
    path.mkdir(parents = True, exist_ok = True)



class Augmentation:
    def __init__(self, src = None, dst = None, src_json = None, auglib = None):
        self.src = src
        self.dst = dst
        
        self.src_json = src_json
        self.dst_json = None
        
        self.dst_images = None
        
        self.auglib = auglib
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
            
            createAugmentation(augmentation, self.src, self.dst_images, self.src_json, self.dst_json, amount)
            copy_image(self.src, self.dst_images)
        except Exception as e:
            print(e)
            traceback.print_exc()
def main():
    augmentation = A.Compose([
        A.RandomCrop(width=640, height=640),
        # A.HorizontalFlip(p=0.5),
        # A.ElasticTransform(always_apply=False, p=1.0, alpha=1.98, sigma=28.86, alpha_affine=10.07, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False),
        # A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, normalized=False)
        # A.RandomBrightnessContrast(p=0.2),

    ])

    src = './images/AP'
    dst = './augmentation'
    src_json = './images/AP.json'
    amount = 10
    
    
    augment = Augmentation()
    augment.setFolder(src, dst)
    augment.setSrcjson(src_json)
    augment.setAugLib('albumentations')
    
    augment.run(amount, augmentation)
    

if __name__ == '__main__':
    main()

