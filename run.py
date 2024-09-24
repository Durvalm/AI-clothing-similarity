import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from extract_features import (load_index, save_index, build_index, search_similar_images)

def load_image_paths_from_directory(directory):
    """Load image paths from a given directory."""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add any other formats you want
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(supported_formats)]

def group_images(image_paths, index, color_hist):
    image_groups = []
    visited = set()  # To track images that have been grouped
    for i, img_path in enumerate(image_paths):
        if i not in visited:
            # Find similar images
            matched_indices = search_similar_images(img_path, index, image_paths, color_hist)
            image_groups.append([image_paths[idx] for idx in matched_indices if idx not in visited])
            visited.update(matched_indices)
    return image_groups

# Specify the directory containing your dataset images
image_directory = "/Users/Durval/Developer/AI-image-similarity/clothes_dataset"
image_paths = load_image_paths_from_directory(image_directory)

index_file = 'color_clothes.index'
if os.path.exists(index_file):
    index = load_index(index_file)  # Load the saved index
    color_hist = np.load("histograms.npy")
else:
    index, color_hist = build_index(image_paths)
    save_index(index, color_hist, index_file)  # Save the index to a file

# Group images and save to folders
image_groups = group_images(image_paths, index, color_hist)
