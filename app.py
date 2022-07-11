import os
import numpy as np
import streamlit as st
from PIL import Image


from dog_app import *


def main():
    st.title("Dog Breed Project")
    st.subheader("udacity capstone project")
    st.text("Project Author: thekael99")

    with st.spinner(text='Loading model...'):
        model = resnet_model
    st.success('Model loaded!')

    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image_file is not None:
        with st.spinner(text='Loading image...'):
            img = Image.open(image_file)
            st.text("Your image:")
            st.image(img.resize((224, 224)))
            img = img.convert('RGB')
            if os.path.exists("./images/predict_img.jpg"):
                os.remove("./images/predict_img.jpg")
            img.save("./images/predict_img.jpg")

        if st.button("Detect Now!"):
            with st.spinner(text='In progress...'):
                class_id, breed = breed_algorithm("./images/predict_img.jpg")
                if class_id == 0:
                    st.text(str("Dog: " + breed))
                elif class_id == 1:
                    st.text(str("Human: " + breed))
                else:
                    st.text("Impossible to identify a human or dog!")


if __name__ == "__main__":
    main()
