import cv2
import numpy as np
import os

absolute_path = "/home/GRAPHICS2/hse_ashevchenko/SegNet_caffe"
images_path = os.path.join(absolute_path, "train_images")
masks_path = os.path.join(absolute_path, "train_masks")

train = open(os.path.join(absolute_path, "train.txt"), "w")

for image_name in os.listdir(images_path):
	current_image_path = os.path.join(images_path, image_name)
	current_mask_path = os.path.join(masks_path, image_name.replace("leftImg8bit", "gtFine_labelIds"))

	train.write("{} {} ".format(current_image_path, current_mask_path))

train.close()
	


