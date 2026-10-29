import numpy as np
import glob
import torch
import cv2
import os
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Relative Path definitions
TRAINING_DATA_SET_PATH = "/keras_slices_data/keras_slices_train"
TRAINING_DATA_SET_PATH_SEG = "/keras_slices_data/keras_slices_seg_train"
TESTING_DATA_SET_PATH = "/keras_slices_data/keras_slices_test"
TESTING_DATA_SET_PATH_SEG = "/keras_slices_data/keras_slices_seg_train"
VALIDATION_DATA_SET_PATH = "/keras_slices_data/keras_slices_validate"
VALIDATION_DATA_SET_PATH_SEG = "/keras_slices_data/keras_slices_seg_validate"

#.nii loading
def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
  channels = np . unique ( arr )
  res = np . zeros ( arr . shape + ( len ( channels ) ,) , dtype = dtype )
  for c in channels :
    c = int ( c )
    res [... , c : c +1][ arr == c ] = 1

  return res
#This function only takes in images with the shape of 256,128
#It ignores the images with a shape of 256,144
def load_data_2D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, early_stop=False):

    affines = []

    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    valid_idx = 0

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        # Skip images with the unwanted shape
        if inImage.shape == (256, 144):
            print(f"Skipping {inName}: shape {inImage.shape}")
            continue

        # Skip any other image that does not match the expected dimensions
        if inImage.shape != (rows, cols):
            print(f"Skipping {inName}: unexpected shape {inImage.shape}")
            continue

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[valid_idx, :, :, :] = inImage
        else:
            images[valid_idx, :, :] = inImage

        affines.append(affine)
        valid_idx += 1

        if i > 20 and early_stop:
            break

    # Remove unused entries for skipped images
    images = images[:valid_idx]

    if getAffines:
        return images, affines
    else:
        return images


#TRAINING
# Get all image paths
folder_path = TRAINING_DATA_SET_PATH + "/*.gz"
file_list = glob.glob(folder_path)
array0 = []
for fname in file_list:
  array0.append(fname.replace(TRAINING_DATA_SET_PATH + "/",""))
os.chdir(TRAINING_DATA_SET_PATH)
training_image_array = load_data_2D(array0)

# Get all image paths
folder_path = TRAINING_DATA_SET_PATH_SEG + "/*.gz"
file_list = glob.glob(folder_path)
array0 = []
for fname in file_list:
  array0.append(fname.replace(TRAINING_DATA_SET_PATH + "/",""))
os.chdir(TRAINING_DATA_SET_PATH_SEG)
training_image_array_seg = load_data_2D(array0)

#TESTING
# Get all image paths
folder_path = TESTING_DATA_SET_PATH + "/*.gz"
file_list = glob.glob(folder_path)
array0 = []
for fname in file_list:
  array0.append(fname.replace(TESTING_DATA_SET_PATH + "/",""))
os.chdir(TESTING_DATA_SET_PATH)
testing_image_array = load_data_2D(array0)

# Get all image paths
folder_path = TESTING_DATA_SET_PATH_SEG + "/*.gz"
file_list = glob.glob(folder_path)
array0 = []
for fname in file_list:
  array0.append(fname.replace(TESTING_DATA_SET_PATH_SEG + "/",""))
os.chdir(TESTING_DATA_SET_PATH_SEG)
testing_image_array_seg = load_data_2D(array0)

#VALIDATION
# Get all image paths
folder_path = VALIDATION_DATA_SET_PATH + "/*.gz"
file_list = glob.glob(folder_path)

array0 = []
for fname in file_list:
  array0.append(fname.replace(VALIDATION_DATA_SET_PATH + "/",""))
os.chdir(VALIDATION_DATA_SET_PATH)
validation_image_array = load_data_2D(array0)

# Get all image paths
folder_path = VALIDATION_DATA_SET_PATH_SEG + "/*.gz"
file_list = glob.glob(folder_path)

array0 = []
for fname in file_list:
  array0.append(fname.replace(VALIDATION_DATA_SET_PATH_SEG + "/",""))
os.chdir(VALIDATION_DATA_SET_PATH_SEG)
validation_image_array_seg = load_data_2D(array0)

#NORMALIZE
for image in training_image_array:
  image = cv2.normalize(
    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
for image in training_image_array_seg:
  image = cv2.normalize(
  image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

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

torch.save(trainloader, "trainloader.pth")
torch.save(testloader,"testloader.pth")
torch.save(validationloader, "validationloader.pth")