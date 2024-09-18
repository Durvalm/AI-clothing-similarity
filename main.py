import cv2
import os
import matplotlib.pyplot as plt

from feature_extraction import (load_index, save_index, build_index, search_similar_images)

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
image_directory = 'clothes_dataset'  # Update this path
image_paths = load_image_paths_from_directory(image_directory)

index_file = 'clothes.index'
if os.path.exists(index_file):
    index = load_index(index_file)  # Load the saved index
else:
    index = build_index(image_paths)
    save_index(index, index_file)  # Save the index to a file

# Provide an input image for searching
input_image_path = 'clothes_dataset/4a56df99-d34f-463c-8c55-7b23dcda50f5.jpg'  # Path to your input image

# Search for similar images
distances, indices = search_similar_images(input_image_path, index, image_paths)

# Show the results
show_images(image_paths, indices)
