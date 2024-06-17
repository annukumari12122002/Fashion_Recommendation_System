import os
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.spatial.distance import cosine

# Load VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image
    if img.mode != "RGB":
        img = img.convert("RGB")  # Convert grayscale to RGB
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input
    return img_array

# Function to recommend fashion items
def recommend_fashion_items(input_image, all_image_paths, model, top_n=5):
    # Pre-process the input image
    preprocessed_img = preprocess_image(input_image)

    # Get features for input image
    input_features = model.predict(preprocessed_img)
    input_features = input_features.flatten() / np.linalg.norm(input_features.flatten())

    # Calculate cosine similarity with all images
    similarities = []
    for img_path in all_image_paths:
        img = Image.open(img_path)
        img_array = preprocess_image(img)
        features = model.predict(img_array)
        features = features.flatten() / np.linalg.norm(features.flatten())
        similarity = 1 - cosine(input_features, features)
        similarities.append((img_path, similarity))

    # Sort by similarity and get top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_images = similarities[:top_n]

    return top_similar_images

# Main function to run the app
def main():
    st.title("Fashion Item Recommendation")

    # Get all image paths
    image_directory = 'women fashion'  # Path to your images directory
    image_paths_list = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Recommendation logic
        if st.button('Recommend Similar Fashion Items'):
            # Recommend similar items
            similar_items = recommend_fashion_items(image, image_paths_list, model, top_n=4)

            # Display recommended images in line
            col1, col2, col3, col4 = st.columns(4)

            for i, (item_path, _) in enumerate(similar_items, start=1):
                item_image = Image.open(item_path)
                with eval(f"col{i}"):
                    st.image(item_image, caption=f"Recommendation {i}", use_column_width=True)

    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    main()
