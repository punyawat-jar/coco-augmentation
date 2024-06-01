from datetime import datetime
from pycocotools.coco import COCO
import json
import random
from fastprogress.fastprogress import master_bar, progress_bar
import os
import gc
import cv2
import albumentations as A
# os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
from module.file_op import *
from module.mask import *

def createAugmentation(augmentation, src_images, dst_images, dst_json, augDesc, unique, amount = 1):
    #Create dir
    coco = COCO(dst_json)
    
    #pick random image
    aug_list = randomPickImage(coco, amount, src_images, unique)
    
    
    for key in progress_bar(aug_list):
        try:
            coco_mask = []
            coco_anns = []
            file_name, image = load_image_coco(coco, key, src_images)
            file_name = f'{augDesc}-aug-{random.choice(range(0,999999))}-{random.choice(range(0,9999999))}-{file_name}'
            
            corr = read_corr_coco(coco, key)
            
            masks, anns = getMask(image, corr)
            
            image_augmented, mask_augmented = augmentingImage(image, masks, augmentation)
            
            for mask in mask_augmented:             # check the mask is empty
                new_coor = mask2polygon(mask)
                coco_mask.append(new_coor)

            for i, mask in enumerate(coco_mask):    # choose the label that have in masks
                if mask != []:
                    coco_anns.append(anns[i])
       
            coco_mask = processMasklist(coco_mask)

            last_imgs_id = CreateAugCOCOAnnotation(coco_mask, dst_json, coco_anns)
            CreateAugCOCOimage(image_augmented, file_name, dst_json, dst_images, last_imgs_id)

            del coco_mask
            del image
            del masks
            del image_augmented
            del mask_augmented
            del new_coor
            del coco_mask
            
            gc.collect()
        except Exception as e:
            continue
        
def CreateAugCOCOAnnotation(coco_mask, dst_json, coco_anns):
        
    with open(dst_json, 'r') as file:
        data = json.load(file)

    # coco = COCO(dst_json)

    # last_imgs_id = list(data.imgs.keys())[-1] + 1
    # last_id = list(data.anns.keys())[-1]
    
    last_imgs_id = data['images'][-1]['id'] + 1
    last_id = data['annotations'][-1]['id']
    for (mask, category_id) in zip(coco_mask, coco_anns):
        last_id += 1
        anno = {
            
                'id': last_id,
                'image_id': last_imgs_id,
                'category_id': category_id,                       # Process to only detect the spine
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
        
    if os.path.exists(img_file):
        image_name = f'aug-{random.choice(range(0,999999))}-{random.choice(range(0,9999999))}-{image_name}'
        img_file = f'{dst_images}/{image_name}'
    else:
        cv2.imwrite(img_file, image_augmented)
    
    

def augmentingImage(image, masks, augmentation):

    augmented = augmentation(image=image, masks=masks)
    
    image_augmented = augmented['image']
    mask_augmented = augmented['masks']
    
    return image_augmented, mask_augmented


def getAugmentlist():
    augDesc = ['HorizontalFlip', 'GaussianBlur', 'GaussNoise', 'CenterCrop', 'InvertImg', 'SafeRotate', 'Superpixels', 'Emboss']
    amountlist = [50, 50, 50, 50, 50, 50, 50, 10]
    unique = [True, True, True, True, True, False, False, True]

    aug1 = A.Compose([
        A.HorizontalFlip(always_apply=True, p=1.0)
    ])
    
    aug2 = A.Compose([
        A.GaussianBlur(always_apply=True, p=1.0, blur_limit=(3, 5), sigma_limit=(0.5, 1.5)) 
    ])

    aug3 = A.Compose([
        A.GaussNoise(always_apply=True, p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0.0)
    ])

    aug4 = A.Compose([
        A.CenterCrop(always_apply=True, p=1.0, height=1000, width=1000)
    ])
    
    aug5 = A.Compose([
        A.InvertImg(always_apply=True, p=1.0)
    ])
    aug6 = A.Compose([
        A.SafeRotate(always_apply=True, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ])

    aug7 = A.Compose([
        A.Superpixels(always_apply=True, p=1.0, p_replace=(0.1, 0.1), n_segments=(100, 100), max_size=128, interpolation=0)
    ])

    aug8 = A.Compose([
        A.Emboss(always_apply=True, p=1.0, alpha=(0.2, 0.5), strength=(0.2, 6.04))
    ])
    
    auglist = [aug1, aug2, aug3, aug4, aug5, aug6, aug7, aug8]
    
    return augDesc, amountlist, unique, auglist