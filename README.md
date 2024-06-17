# Fashion Item Recommendation App

This is a Streamlit app for recommending similar fashion items based on an uploaded image.

## Installation

1. **Set up Conda Virtual Environment**

```bash
conda create -n frs python=3.7.6
conda activate frs

````
2.  **Installation of Packages** 

````
pip install -r requirements.txt

````

3. Usage
Run the app using the following command:

````
streamlit run app.py

````

4 . Conclusion 

We proposed a novel fashion recommendation framework leveraging VGG19, consisting of two stages: feature extraction using the VGG19 CNN classifier and generating similar fashion product images based on the input image. This approach enhances recommendation accuracy and improves the fashion exploration experience for users by allowing them to upload any fashion image.