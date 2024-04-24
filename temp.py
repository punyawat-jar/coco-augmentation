import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import albumentations as A
import shutil
import json
from multiprocessing import Pool, Queue
import time
from fastprogress.fastprogress import master_bar, progress_bar
import traceback

from module.file_op import *
from module.display import *
from module.mask import *
from module.augment import *

class Augmentation:
    def __init__(self, src=None, dst=None, src_json=None, queue=None):
        self.src = src
        self.dst = dst
        self.src_json = src_json
        self.dst_json = None
        self.dst_images = None
        self.amount = 1
        self.queue = queue  # Queue for progress updates

    def setFolder(self, src, dst):
        self.src = src
        self.dst = dst

    def setSrcjson(self, src_json):
        self.src_json = src_json

    def setAugLib(self, auglib):
        self.auglib = auglib

    def run(self, amount, augmentation, process_id):
        try:
            self.dst_images = f'{self.dst}/images'
            self.dst_json = f'{self.dst}/data_{process_id}.json'  # unique JSON for each process
            self.amount = amount

            if not os.path.exists(self.src_json):
                raise Exception(f"Source file not found: {self.src_json}")

            cleanDst(self.dst, self.src_json, self.dst_json)

            mkdir(self.dst)
            mkdir(self.dst_images)
            shutil.copyfile(self.src_json, self.dst_json)

            createAugmentation(augmentation, self.src, self.dst_images, self.dst_json, amount, self.queue)
        except Exception as e:
            print(e)
            traceback.print_exc()

def combine_json_files(data):
    combined_data = {'images': [], 'annotations': [], 'categories': []}  # Assuming structure
    next_image_id = 0
    next_annotation_id = 0
    
    for d in data:
        json_path = f'./augmentation/{d}/data_{d}.json'
        with open(json_path, 'r') as file:
            part_data = json.load(file)
            
            # Adjust image ids and annotation ids
            image_id_mapping = {}
            for image in part_data['images']:
                old_id = image['id']
                new_id = next_image_id
                image_id_mapping[old_id] = new_id
                image['id'] = new_id
                combined_data['images'].append(image)
                next_image_id += 1
            
            for annotation in part_data['annotations']:
                old_image_id = annotation['image_id']
                new_image_id = image_id_mapping[old_image_id]
                annotation['image_id'] = new_image_id
                annotation['id'] = next_annotation_id
                combined_data['annotations'].append(annotation)
                next_annotation_id += 1
            
            # Assuming categories are consistent and only need to be copied once
            if 'categories' in part_data and not combined_data['categories']:
                combined_data['categories'] = part_data['categories']
    
    # Write combined data to a new JSON file
    with open('./augmentation/combined_data.json', 'w') as file:
        json.dump(combined_data, file, indent=4)

    # Optionally, cleanup individual JSON files
    for d in data:
        os.remove(f'./augmentation/{d}/data_{d}.json')

def process_data(i, amount, augmentation, queue):
    src = f'../../dataset/split/{i}/train'
    dst = f'./augmentation/{i}'
    src_json = f'../../dataset/{i}.json'

    augment = Augmentation(src, dst, src_json, queue)
    augment.setFolder(src, dst)
    augment.setSrcjson(src_json)
    augment.run(amount, augmentation, i)

def main():
    data = ['AP']  # Modify as needed
    amount = 2550
    augmentation = A.Compose([
        A.RandomCrop(always_apply=False, width=640, height=640, p=0.2),
        A.SafeRotate(always_apply=False, p=0.2, limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
        A.PixelDropout(always_apply=False, p=0.1, dropout_prob=0.26, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None),
        A.ElasticTransform(always_apply=False, p=0.2, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False)
    ])

    queue = Queue()  # Queue for inter-process communication
    mb = master_bar(data)
    
    with Pool(processes=len(data)) as pool:
        for i in mb:
            pool.apply_async(process_data, (i, amount, augmentation, queue))
        
            pb = progress_bar(range(amount), parent=mb)
            for _ in pb:
                # Receive progress updates from each subprocess
                progress = queue.get()
                if progress is not None:
                    pb.update(progress)

    # Combine JSON files here after processing
    combine_json_files(data)

if __name__ == '__main__':
    main()
