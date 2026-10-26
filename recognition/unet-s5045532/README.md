# Recognition Tasks
- Image segmentation.

This project uses a 2D Improved-UNet for image segmentation of the keras_slices_data dataset which is based in hip MRI's.

# Feasiblity report (from a word doc)

Intended user: The intended user is for medical staff. 

Model: 

2D-improved UNet for image segmentation. 

The plan is to get a Standard global Sørensen-Dice similarity coefficient greater than 0.75. 

We will start off with a standard 2D U-net and then add in shortcuts and other characteristics that define it as a 2D-Improved U-net. This will give us a performance benchmark. 

Dataset: 

HipMRI 2D slices (keras_slices_data). 

The total size of the dataset will be 11338 files. This will be split into 1120 files for validation, 9664 files for training and 554 files for testing. 

Images are 256 x 256 and are black and white (1 channel). We will be putting it in batches of 4. So what we'll have is 4x1x256x256 for dimensions. We can then encode this down from 256x256 down to 1 and then back up to 256x256. 

We will be using Nifti loading to preserve exact spacial alignment between images and ground truth masks. We will enforce strict paitent level splitting of the dataset. We will use intensity normalization and ground truth masks.  

Difficulty band: Normal, 

Course concepts: This is dealing with the bread and butter of the course since we are working with medical imaging training, image segmentation and Demo 2 Task 4 Medium difficulty has us working with U-Net. 

Since we have dealt with U-Net and the keras_slices_data dataset before I don’t think it will cost too many resources and my current plan is to try and do as much as I can from google collab and train the model on rangpur when necessary. We will benchmark the model by looking at the per-class Sørensen-Dice similarity coefficients and the Intersection-over-Union on the test split. 

Workable fallback: If the Improved U-Net isn’t working out, I can fall back to the 2D-U-net (on the easy difficulty band). 

Key technical risks would be patient data leaking. 

Computing budget/Resource constraint: The computing budget when using google collab is 12.7 GB of RAM, 107GB of disc space and a T4 GPU with 16GB of VRAM. Alternatively, we can use rangpur where we have access to a NVIDA A100 GPU with 40GB or 80GB of VRAM and 7.5GB of disc space but limited usage since it's shared with the other members of the course. 

# Working principles
To make sure that I document all AI usage involved in the project.

# Instructions
Place your keras_slices_data .zip folder in the unet-s5045532, the python script will automatically unzip the folder and start using it to train/test/validate.

# Artificial Intelligence Usage Disclosure
- Which AI tool was used: Chat GPT for code generation.

Chat GPT was used to generate the model, training, testing and verification code for the model. It was also used to generate some of the data loading script.

I audited the code by first checking if there was any issues with runtime (e.g. exceptions thrown by the torch functions). I then checked the output of the model by giving the model a image, getting the output and plotting it to see if the output was consistent with what was required of the model.
