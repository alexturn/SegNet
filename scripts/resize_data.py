import cv2
import numpy as np
import os

absolute_path = "/home/GRAPHICS2/hse_ashevchenko/SegNet"
images_path = os.path.join(absolute_path, "train_images")
masks_path = os.path.join(absolute_path, "train_masks")

path_to_save_image = "/home/GRAPHICS2/hse_ashevchenko/SegNet_caffe/train_images"
path_to_save_mask = "/home/GRAPHICS2/hse_ashevchenko/SegNet_caffe/train_masks"

for image_name in os.listdir(images_path):
	current_image_path = os.path.join(images_path, image_name)
	current_mask_path = os.path.join(masks_path, image_name.replace("leftImg8bit", "gtFine_labelIds"))

	mask = cv2.resize(cv2.imread(current_mask_path), 
                                (448,224), interpolation=cv2.INTER_NEAREST)[:,:,0]

	image = cv2.resize(cv2.imread(current_image_path),
		                         (448,224))

	cv2.imwrite(os.path.join(path_to_save_image, image_name), image)
	cv2.imwrite(os.path.join(path_to_save_mask, image_name.replace("leftImg8bit", "gtFine_labelIds")), mask)
