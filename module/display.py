import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def showImage(coco, randimg, increase_size =2):
    img = cv2.imread(randimg)
    
    for i in coco.imgs:
        if coco.imgs.get(i)['file_name'] == randimg.split('/')[-1]:
            image_num = i
    masks = []
    for i in coco.anns:
        if coco.anns.get(i)['image_id'] == image_num:
            for item in coco.anns.get(i)['segmentation']:
                masks.append(item)
                    
    listMask = []
    mask = np.array([])
    mask_img = Image.new('L', (img.shape[1], img.shape[0]), 0)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    draw = ImageDraw.Draw(mask_img)
    
    for polygon in masks:
        # print(polygon)
        draw.polygon(polygon, outline = 1, fill= 1)
        # draw.point(polygon, fill=1)
        mask = np.array(mask_img)
        
        listMask.append(mask)
        

    
    plt.figure(figsize= increase_size * np.array(plt.rcParams['figure.figsize']))

    plt.imshow(image, cmap='gray')

    plt.imshow(mask, alpha=0.2) 

    plt.show()

    return masks