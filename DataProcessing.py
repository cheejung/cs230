# This script converts the EuroSAT dataset to a hybrid dataset and respective high-/low-res test sets
import os
import random
import numpy as np
from PIL import Image
from skimage import util, filters, transform
from scipy import linalg as la
import shutil

# Custom functions to add noise
def rank_k_approx(image_matrix, k):
    approx_matrix = np.zeros(image_matrix.shape)
    for i in range(image_matrix.shape[-1]):
        U, s, Vh = la.svd(image_matrix[...,i], full_matrices=False)
        approx_matrix[...,i] = np.dot(U[:,:k]*s[:k], Vh[:k,:])
    return approx_matrix

def add_noise(image_matrix, mean=0, std_dev=0.1):
    noisy_image = util.random_noise(image_matrix, mode='gaussian', mean=mean, var=std_dev**2, clip=True)
    return noisy_image

def add_blur(image_matrix, sigma=1):
    blurred_image = filters.gaussian(image_matrix, sigma=sigma, channel_axis=-1)
    return blurred_image

def add_rotation(image_matrix, angle):
    rotated_image = transform.rotate(image_matrix, angle=angle, mode='edge', preserve_range=True)
    return rotated_image

# Given high-res dataset, for *each* class (or folder), we do a 65/35 split. 65% of images will be processed, 
# and 35% of images will simply be kept the same. These images are then moved a folder called "Hybrid"

# Path to EuroSAT
path_eurosat = '/Users/heejungchoi/Desktop/EuroSAT'
path_hybrid = '/Users/heejungchoi/Desktop/Hybrid'

# Create Hybrid directory
os.makedirs(path_hybrid, exist_ok=True)

# List of classes
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
           'Highway', 'Industrial', 'Pasture', 
           'PermanentCrop', 'Residential', 'River', 'SeaLake']

for class_name in classes:
    # Path to specific class
    class_path = os.path.join(path_eurosat, class_name)
    
    # Path to specific class in Hybrid
    hybrid_class_path = os.path.join(path_hybrid, class_name)
    
    # Create directory in Hybrid
    os.makedirs(hybrid_class_path, exist_ok=True)
    
    # We List all the images in the class directory
    images = os.listdir(class_path)
    
    # Shuffle the list of images
    random.shuffle(images)
    
    # Split images into original and to be processed
    n_images = len(images)
    split_point = int(0.35 * n_images)
    
    original_images = images[:split_point]
    to_be_processed_images = images[split_point:]
    
    # Copy original images to Hybrid
    for image_name in original_images:
        image_path = os.path.join(class_path, image_name)
        new_image_path = os.path.join(hybrid_class_path, image_name)
        
        image = Image.open(image_path)
        
        image.save(new_image_path)
        
    # Process the remaining images
    for image_name in to_be_processed_images:
        image_path = os.path.join(class_path, image_name)
        new_image_path = os.path.join(hybrid_class_path, image_name)
        
        # Open image and convert to matrix
        image = Image.open(image_path)
        image_matrix = np.array(image)
        
        # Normalize image
        image_matrix = image_matrix / 255.0

        # Perform rank-K approximation and add noise, blur, and rotation
        k = np.random.randint(15, 50)
        
        approx_matrix = rank_k_approx(image_matrix, k)
        noisy_and_blurred = add_noise(add_blur(approx_matrix, sigma=1), mean=0, std_dev=0.01)
        processed_matrix = add_rotation(noisy_and_blurred, angle=np.random.randint(0, 25))

        # Ensure image pixel values are within appropriate range
        processed_matrix = np.clip(processed_matrix, 0, 1)

        # Convert back to image
        processed_image = Image.fromarray((processed_matrix * 255).astype(np.uint8))

        # Save processed image
        processed_image.save(new_image_path)

# Now, in the Hybrid folder, we move 70% of images in each class to train folder, 20% to val folder, 
# and 10% test folder

# Define the split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

# We now create HybridTrain, HybridVal and HybridTest directories (folders)
path_hybrid_train = '/Users/heejungchoi/Desktop/HybridTrain'
path_hybrid_val = '/Users/heejungchoi/Desktop/HybridVal'
path_hybrid_test = '/Users/heejungchoi/Desktop/HybridTest'

os.makedirs(path_hybrid_train, exist_ok=True)
os.makedirs(path_hybrid_val, exist_ok=True)
os.makedirs(path_hybrid_test, exist_ok=True)

# List of the classes
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 
           'PermanentCrop', 'Residential', 'River', 'SeaLake']

for class_name in classes:
    # The path to specific class in Hybrid
    hybrid_class_path = os.path.join(path_hybrid, class_name)
    
    # Path to specific class in HybridTrain, HybridVal and HybridTest
    hybrid_train_class_path = os.path.join(path_hybrid_train, class_name)
    hybrid_val_class_path = os.path.join(path_hybrid_val, class_name)
    hybrid_test_class_path = os.path.join(path_hybrid_test, class_name)
    
    # We make the directories in HybridTrain, HybridVal and HybridTest
    os.makedirs(hybrid_train_class_path, exist_ok=True)
    os.makedirs(hybrid_val_class_path, exist_ok=True)
    os.makedirs(hybrid_test_class_path, exist_ok=True)
    
    # List all images in the class directory
    images = os.listdir(hybrid_class_path)
    
    # Shuffle the list once more to get a good distribution
    random.shuffle(images)
    
    # Determine the split points
    n_images = len(images)
    train_point = int(train_split * n_images)
    val_point = int(val_split * n_images) + train_point
    
    train_images = images[:train_point]
    val_images = images[train_point:val_point]
    test_images = images[val_point:]
    
    # Transfer images to HybridTrain, HybridVal and HybridTest
    for image_name in train_images:
        image_path = os.path.join(hybrid_class_path, image_name)
        new_image_path = os.path.join(hybrid_train_class_path, image_name)
        
        shutil.move(image_path, new_image_path)
        
    for image_name in val_images:
        image_path = os.path.join(hybrid_class_path, image_name)
        new_image_path = os.path.join(hybrid_val_class_path, image_name)
        
        shutil.move(image_path, new_image_path)
        
    for image_name in test_images:
        image_path = os.path.join(hybrid_class_path, image_name)
        new_image_path = os.path.join(hybrid_test_class_path, image_name)
        
        shutil.move(image_path, new_image_path)

# Make a HighResTest set

# Paths
path_highres_test = '/Users/heejungchoi/Desktop/HighResTest'
path_eurosat = '/Users/heejungchoi/Desktop/EuroSAT'
path_hybrid_test = '/Users/heejungchoi/Desktop/HybridTest'


# Create HighResTest directory
os.makedirs(path_highres_test, exist_ok=True)

# List of the classes
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 
           'PermanentCrop', 'Residential', 'River', 'SeaLake']

for class_name in classes:
    # Path to specific class in EuroSAT and HybridTest
    eurosat_class_path = os.path.join(path_eurosat, class_name)
    hybrid_test_class_path = os.path.join(path_hybrid_test, class_name)
    
    # Path to specific class in HighResTest
    highres_test_class_path = os.path.join(path_highres_test, class_name)
    
    # Create directory in HighResTest
    os.makedirs(highres_test_class_path, exist_ok=True)
    
    # List all images in the class directory in HybridTest
    images = os.listdir(hybrid_test_class_path)
    
    # For each image in HybridTest, get the corresponding image in EuroSAT and copy to HighResTest
    for image_name in images:
        image_path = os.path.join(eurosat_class_path, image_name)
        new_image_path = os.path.join(highres_test_class_path, image_name)
        
        shutil.copy(image_path, new_image_path)

# Make a Lowres test set

# Define the path to LowResTest directory
path_lowres_test = '/Users/heejungchoi/Desktop/LowResTest'

# Create LowResTest directory
os.makedirs(path_lowres_test, exist_ok=True)

for class_name in classes:
    # Path to specific class in EuroSAT and HybridTest
    eurosat_class_path = os.path.join(path_eurosat, class_name)
    hybrid_test_class_path = os.path.join(path_hybrid_test, class_name)
    
    # Path to specific class in LowResTest
    lowres_test_class_path = os.path.join(path_lowres_test, class_name)
    
    # Create directory in LowResTest
    os.makedirs(lowres_test_class_path, exist_ok=True)
    
    # List all images in the class directory in HybridTest
    images = os.listdir(hybrid_test_class_path)
    
    # For each image in HybridTest, get the corresponding image in EuroSAT, process it, and copy to LowResTest
    for image_name in images:
        image_path = os.path.join(eurosat_class_path, image_name)
        new_image_path = os.path.join(lowres_test_class_path, image_name)
        
        # Open image and convert to matrix
        image = Image.open(image_path)
        image_matrix = np.array(image)

        # Normalize image
        image_matrix = image_matrix / 255.0

        # Perform rank-K approximation and add noise, blur, and rotation
        k = np.random.randint(15, 50)
        
        approx_matrix = rank_k_approx(image_matrix, k)
        noisy_and_blurred = add_noise(add_blur(approx_matrix, sigma=1), mean=0, std_dev=0.01)
        processed_matrix = add_rotation(noisy_and_blurred, angle=np.random.randint(0, 25))

        # Ensure image pixel values are within appropriate range
        processed_matrix = np.clip(processed_matrix, 0, 1)

        # Convert back to image
        processed_image = Image.fromarray((processed_matrix * 255).astype(np.uint8))

        # Save processed image
        processed_image.save(new_image_path)
