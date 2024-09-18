import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import faiss


# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Load the pre-trained model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    """Extract features from an image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def build_index(image_paths):
    """Build FAISS index for a list of image paths."""
    features_list = []
    for img_path in image_paths:
        features = extract_features(img_path)
        features_list.append(features)
    features_array = np.array(features_list).astype('float32')

    # Build FAISS index
    dimension = features_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(features_array)
    return index

def save_index(index, filename):
    """Save the FAISS index to a file."""
    faiss.write_index(index, filename)

def load_index(filename):
    """Load the FAISS index from a file."""
    return faiss.read_index(filename)

def search_similar_images(img_path, index, image_paths, k=10):
    """Search for similar images using the FAISS index."""
    query_features = extract_features(img_path)
    query_features = np.expand_dims(query_features, axis=0).astype('float32')

    # Perform search
    distances, indices = index.search(query_features, k)
    return distances, indices



