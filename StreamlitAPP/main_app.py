#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf


model=load_model('plant_disease_model_1.h5')

class_names=['Corn(maize)_CommonRust','Gray_LeafSpot','Healthy_no problem ']

st.title("plant leaf disease detection")
st.markdown("upload an image of the leaf")
plant_image=st.file_uploader("choose an image...",type="jpg")
submit=st.button("predict disease")
if submit:
  if plant_image is not None:
    file_bytes=np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
    opencv_image=cv2.imdecode(file_bytes,1)
    st.image(opencv_image)
    st.write(opencv_image.shape)
    opencv_image=cv2.resize(opencv_image,(256,256))
    opencv_image.shape=(1,256,256,3)
    y_pred=model.predict(opencv_image)
    result=class_names[np.argmax(y_pred)]
    st.title(str("This is "+ result.split('_')[0]+"leaf with "+result.split('_')[1]))

