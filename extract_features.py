import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import faiss
from segment import segment_image


# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_color_features(img_array):
    hist = cv2.calcHist([img_array], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(img_array):
    """Extract only ResNet features from an image."""
    if img_array.shape[:2] != (224, 224):
        img_array = cv2.resize(img_array, (224, 224))
    img_array_prep = preprocess_input(np.expand_dims(img_array, axis=0))
    resnet_features = model.predict(img_array_prep).flatten()
    return resnet_features


def build_index(image_paths):
    """Build FAISS index for a list of image paths."""
    features_list = []
    color_histograms = []
    for img_path in image_paths:
        print(img_path)
        segmented_img = segment_image(img_path)
        features = extract_features(segmented_img)
        features_list.append(features)

        color_hist = extract_color_features(segmented_img)
        color_histograms.append(color_hist)

    features_array = np.array(features_list).astype('float32')

    # Build FAISS index
    dimension = features_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features_array)
    return index, np.array(color_histograms)

def save_index(index, histograms, filename):
    """Save the FAISS index to a file."""
    faiss.write_index(index, filename)
    np.save("histograms.npy", histograms)

def load_index(filename):
    """Load the FAISS index from a file."""
    return faiss.read_index(filename)


def search_similar_images(img_path, index, image_paths, color_histograms, k=10):
    """Search for similar images using the FAISS index and filter by color histogram similarity."""
    segmented_img = segment_image(img_path)
    query_features = extract_features(segmented_img)
    query_features = np.expand_dims(query_features, axis=0).astype('float32')
    
    # Perform search with ResNet features to find top k results
    distances, indices = index.search(query_features, k)
    # Extract color histogram for the query image
    query_color_hist = extract_color_features(segmented_img)
    
    # Filter results by color similarity
    final_matches = []
    for i, idx in enumerate(indices[0]):
        candidate_color_hist = color_histograms[idx]
        # Using Bhattacharyya distance
        color_distance = cv2.compareHist(query_color_hist, candidate_color_hist, cv2.HISTCMP_BHATTACHARYYA)
        resnet_distance = distances[0][i]
        
        # Lower distance indicates more similarity; define a strict threshold for "very similar" colors
        if color_distance < 0.16 and resnet_distance < 800:  
            final_matches.append(idx)

    return final_matches