import numpy as np
import glob
import numpy as np
from PIL import Image
import torch
import numpy as np
from PIL import Image
import cv2

TRAINING_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_train"
TRAINING_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_train"
TESTING_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_test"
TESTING_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_train"
VALIDATION_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_validate"
VALIDATION_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_validate"


!unzip /content/keras_png_slices_data.zip -d /content/

TRAINING_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_train"
TRAINING_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_train"
TESTING_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_test"
TESTING_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_train"
VALIDATION_DATA_SET_PATH = "/content/keras_png_slices_data/keras_png_slices_validate"
VALIDATION_DATA_SET_PATH_SEG = "/content/keras_png_slices_data/keras_png_slices_seg_validate"


import os
import glob
import numpy as np
from PIL import Image
import cv2
#TRAINING
# Get all image paths (e.g., .png files)
folder_path = TRAINING_DATA_SET_PATH + "/*.png"
file_list = glob.glob(folder_path)

# Load each image as an array and combine them
training_image_array = np.array([np.array(Image.open(fname)) for fname in file_list])

# Get all image paths (e.g., .png files)
folder_path = TRAINING_DATA_SET_PATH_SEG + "/*.png"
file_list = glob.glob(folder_path)
print(training_image_array[0])
# Load each image as an array and combine them
training_image_array_seg = np.array([np.array(Image.open(fname)) for fname in file_list])
#TESTING
# Get all image paths (e.g., .png files)
folder_path = TESTING_DATA_SET_PATH + "/*.png"
file_list = glob.glob(folder_path)

# Load each image as an array and combine them
testing_image_array = np.array([np.array(Image.open(fname)) for fname in file_list])

# Get all image paths (e.g., .png files)
folder_path = TESTING_DATA_SET_PATH_SEG + "/*.png"
file_list = glob.glob(folder_path)

# Load each image as an array and combine them
testing_image_array_seg = np.array([np.array(Image.open(fname)) for fname in file_list])

#VALIDATION
# Get all image paths (e.g., .png files)
folder_path = VALIDATION_DATA_SET_PATH + "/*.png"
file_list = glob.glob(folder_path)

# Load each image as an array and combine them
validation_image_array = np.array([np.array(Image.open(fname)) for fname in file_list])

folder_path = VALIDATION_DATA_SET_PATH_SEG + "/*.png"
file_list = glob.glob(folder_path)

# Load each image as an array and combine them
validation_image_array_seg = np.array([np.array(Image.open(fname)) for fname in file_list])

#NORMALIZE
for image in training_image_array:
  image = cv2.normalize(
    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
for image in training_image_array_seg:
  image = cv2.normalize(
  image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class CustomImageDataset(Dataset):
  imageSet : np.ndarray
  segImageSet : np.ndarray
  def __init__(self, imageSet, segImageSet):
    self.imageSet = imageSet
    self.segImageSet = segImageSet
  def __len__(self):
    return len(self.imageSet)
  def __getitem__(self, index):
     return (self.imageSet[index],self.segImageSet[index])
trainingDataset = CustomImageDataset(training_image_array,training_image_array_seg)
testingDataset = CustomImageDataset(testing_image_array,testing_image_array_seg)
validationDataset = CustomImageDataset(validation_image_array,validation_image_array_seg)

batch_size = 64

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testingDataset, batch_size=batch_size, shuffle=False, num_workers=2)
validationloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=False, num_workers=2)