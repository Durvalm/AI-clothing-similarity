import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from extract_features import (load_index, save_index, build_index, search_similar_images)

def show_images(image_paths, indices):
    """Display images based on the indices retrieved from the search."""
    fig, axes = plt.subplots(1, len(indices[0]), figsize=(15, 5))
    for i, idx in enumerate(indices[0]):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()


def load_image_paths_from_directory(directory):
    """Load image paths from a given directory."""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add any other formats you want
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(supported_formats)]


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

# Provide an input image for searching
# input_image_path = '/Users/Durval/Downloads/img.jpg'  # Path to your input image
input_image_path = "/Users/Durval/Developer/AI-image-similarity/clothes_dataset/IMG_9827.JPG"

# Search for similar images
matches = search_similar_images(input_image_path, index, image_paths, color_hist)
matched_image_paths = [img_path for img_path, *_ in matches]

# Function to display images
def show_images(image_paths):
    num_images = len(image_paths)
    cols = 3  # Number of columns
    rows = (num_images // cols) + (num_images % cols > 0)  # Calculate number of rows

    plt.figure(figsize=(15, 5 * rows))  # Adjust the figure size based on rows
    for i, img_path in enumerate(image_paths):
        img = plt.imread(img_path)
        plt.subplot(rows, cols, i + 1)  # Create subplot with dynamic rows
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Match {i + 1}")
    plt.tight_layout()
    plt.show()

# Show the results
show_images(matched_image_paths)

