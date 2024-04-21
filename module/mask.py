
import cv2
import numpy as np
from PIL import Image, ImageDraw
        

def mask2bbox(segmentation):
    points = np.array(segmentation, dtype=np.float32).reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(points)
    return [x, y, w, h]


def maskArea(polygons):
    if len(polygons) % 2 != 0:
        raise ValueError("The polygon points list must contain an even number of values.")
    area = 0.0

    num_points = len(polygons) // 2

    for i in range(0, num_points*2, 2):
        x1, y1 = polygons[i % (num_points*2)], polygons[(i + 1) % (num_points*2)]
        x2, y2 = polygons[(i + 2) % (num_points*2)], polygons[(i + 3) % (num_points*2)]
        area += x1 * y2 - y1 * x2

    area = abs(area) / 2.0
    return area


def getMask(image, corr):
    masks = []
    
    for polygon in corr:
        mask_img = Image.new('L', (image.shape[1], image.shape[0]), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(polygon['segmentation'][0], fill=1)
        mask = np.array(mask_img)
        masks.append(mask)
    return masks


def processMasklist(coco_mask):
    flag = 0
    for i in range(0, len(coco_mask) - 1):
        if coco_mask[i] == coco_mask[i+1]:
            flag = i
            break
        
    if flag != 0:
        for i in reversed(range(i+1, len(coco_mask))):
            coco_mask.pop(i)

    for i in reversed(range(len(coco_mask))):
        if coco_mask[i] == []:
            coco_mask.pop(i)
    return coco_mask


def mask2polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 5:
            segmentation.append(contour)

    return segmentation

