from datetime import datetime
from pycocotools.coco import COCO
import json
import random
from tqdm import tqdm

import cv2
from module.file_op import *
from module.mask import *

def createAugmentation(augmentation, src_images, dst_images, dst_json, amount = 1):
    #Create dir
    coco = COCO(dst_json)
    keys = list(coco.imgs.keys())
    
    #pick random image
    aug_list = randomPickImage(keys, amount)
    
    
    for key in tqdm(aug_list, desc = 'Creating Augmentation'):
        coco_mask = []
        
        file_name, image = load_image_coco(coco, key, src_images)
        file_name = f'aug{random.choice(range(0,100000))}-{file_name}'
        
        corr = read_corr_coco(coco, key)
        
        masks = getMask(image, corr)
        
        image_augmented, mask_augmented = augmentingImage(image, masks, augmentation)
        
        for mask in mask_augmented:
            new_coor = mask2polygon(mask)
            coco_mask.append(new_coor)

        coco_mask = processMasklist(coco_mask)

        last_imgs_id = CreateAugCOCOAnnotation(coco_mask, dst_json)
        CreateAugCOCOimage(image_augmented, file_name, dst_json, dst_images, last_imgs_id)
        
def CreateAugCOCOAnnotation(coco_mask, dst_json):
        
    with open(dst_json, 'r') as file:
        data = json.load(file)

    coco = COCO(dst_json)

    last_imgs_id = list(coco.imgs.keys())[-1] + 1
    print(last_imgs_id)
    last_id = list(coco.anns.keys())[-1]
    print(last_id)
    for mask in coco_mask:
        last_id += 1
        anno = {
            
                'id': last_id,
                'image_id': last_imgs_id,
                'category_id': 0,                       # Process to only detect the spine
                'segmentation': mask,
                'bbox': mask2bbox(mask[0]),
                'area': maskArea(mask[0]),
                'iscrowd': 0
        }
        
        data['annotations'].append(anno)
        
    with open(dst_json,'w') as f:
        json.dump(data, f, indent = 4, default=np_encoder)
        
    return last_imgs_id
    

def CreateAugCOCOimage(image_augmented, image_name, dst_json, dst_images, last_imgs_id):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    with open(dst_json, 'r') as file:
            data = json.load(file)
    imgs = {
            'id': last_imgs_id,
            'file_name': image_name,
            'height': image_augmented.shape[0],
            'width': image_augmented.shape[1],
            'date_captured' : date_time,
            'license': 1
        }
    
    data['images'].append(imgs)
    
    img_file = f'{dst_images}/{image_name}'
    
    with open(dst_json,'w') as f:
        json.dump(data, f, indent = 4, default=np_encoder)
        
    cv2.imwrite(img_file, image_augmented)
    
    

def augmentingImage(image, masks, augmentation):

    augmented = augmentation(image=image, masks=masks)
    
    image_augmented = augmented['image']
    mask_augmented = augmented['masks']
    
    return image_augmented, mask_augmented

