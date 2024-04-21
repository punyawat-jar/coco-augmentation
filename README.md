# coco-augmentation

The ready-to-run image augmentation by using [Albumentations](https://github.com/albumentations-team/albumentations) which **designed to work with the COCO image format.**

## Setting up

The program assumed that you have the image in the COCO format. Therefore, all you have to setting up is config the [create_augmentation.py](https://github.com/punyawat-jar/coco-augmentation/blob/main/create_augmentation.py) parameters.

1. augmentation is the Albumentations's **Compose** class to define an augmentation pipeline. You can adjust as you want, which you can read more about this topic [here](https://albumentations.ai/docs/getting_started/image_augmentation/)
2. src is the source directory where the images located.
3. dst is the destination directory where the source and augmented images will be pasted.
4. src_json is the source json file directory. **It's must be in the COCO format, if not it cannot be processed.**

## Running Code
After config the file, you can run the code by using:

```
python create_augmentation.py
```

## Result
After the processing, you can find the all the source image and the augmented image in the dst directory. The json file will be appened with the annotation after the augmentation. This code is currently perform with the images segmentation task, so if there're any bug, please report.
