from transformers import pipeline
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


st.title('OWL-ViT')

# OWL-ViT
detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

upload_img_file = st.file_uploader(
    'Upload Image', type=['jpg', 'jpeg', 'png'])
FRAME_WINDOW = st.image([])
if upload_img_file is not None:
    file_bytes = np.asarray(
        bytearray(upload_img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img)
    text_prompt = st.text_input('Text Prompt:', value="human face, rocket, nasa badge, star-spangled banner").split(',')
    if st.button('Submit', use_container_width=True):
        st.subheader('Result')
        predictions = detector(
            Image.fromarray(img),
            candidate_labels=text_prompt,
        )
        for i in predictions:
            score, label, box = i['score'], i['label'], i['box']
            plot_one_box(
                [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                img, (0, 0, 255),
                f"{label}: {score:.2f}", 3
            )
        st.image(img)
