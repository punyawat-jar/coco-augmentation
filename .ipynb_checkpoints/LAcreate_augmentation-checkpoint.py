import os
# os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import albumentations as A

import shutil
from fastprogress.fastprogress import master_bar, progress_bar

from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback

from module.file_op import *
from module.display import *
from module.mask import *
from module.augment import *

class Augmentation:
    def __init__(self, src = None, dst = None, src_json = None, augSet = None):
        self.src = src
        self.dst = dst
        self.src_json = src_json
        self.augSet = augSet
        self.dst_json = None
        self.augDesc = None
        self.dst_images = None
        self.amount = 1

    def setFolder(self, src, dst):
        self.src = src
        self.dst = dst

    def setSrcjson(self, src_json):
        self.src_json = src_json
        
    def run(self, amount, augmentation, augSet, augDesc, unique):
        try:
            self.dst_images = f'{self.dst}/images'
            self.dst_json = f'{self.dst}/data{augSet}.json'
            self.amount = amount
            self.augSet = augSet
            self.augDesc = augDesc
            if not os.path.exists(self.src_json):
                raise Exception(f"Source file not found: {self.src_json}")

            shutil.copyfile(self.src_json, self.dst_json)
            
            createAugmentation(augmentation, self.src, self.dst_images, self.dst_json, augDesc, unique, amount)
        except Exception as e:
            print(e)
            traceback.print_exc()

def processAug(amountlist, src, dst, src_json, augSet, j, augDesc, unique):
    augment = Augmentation()
    augment.setFolder(src, dst)
    augment.setSrcjson(src_json)
    
    augment.run(amountlist[j], augSet, j, augDesc[j], unique[j])
    

def main():
    data = ['LA']
    
    augDesc, amountlist, unique, auglist = getAugmentlist()

    num_cpus = cpu_count()
    print(f'All cpu process is {num_cpus}')
    num_cpus = int(num_cpus/2)
    print(f'Using cpu : {num_cpus}')
    
    pool = Pool(num_cpus)
    
    for i in data:
        src = f'../../dataset/split/{i}/train'
        dst = f'./augmentation/{i}'
        src_json = f'../../dataset/{i}.json'
        full_json = f'{dst}/Data.json'
        cleanDst(dst)
        
        mkdir(dst)
        mkdir(f'{dst}/images')

        progress = tqdm(total=len(auglist), desc = 'Processing Augmentation image sets')
            
        
        for j, augSet in enumerate(auglist):
            pool.apply_async(processAug, args = (amountlist, src, dst, src_json, augSet, j, augDesc, unique))
            progress.update()
        progress.refresh()
        pool.close()
        pool.join()
            
    jsonList = glob.glob(f'{dst}/*.json')
    jsonList.sort()
    
    for i in range(len(jsonList)-1):
        pyodi.coco_merge(jsonList[0], jsonList[i+1], jsonList[0])
    
    os.rename(jsonList[0], full_json)
    jsonList.pop(0)
    for file in jsonList:
        os.remove(file)
        print(f"Deleted file: {file}")

    shutil.move(f'{dst}/Data.json', f'{dst}/images/Data.json')
if __name__ == '__main__':
    main()

