import numpy as np
import cv2
from retinex import automatedMSRCR
import streamlit as st
import base64

img = st.file_uploader('image pls:')
gama = st.slider('gama', 0.0, 10.0, 1.0, 0.1)
if st.button('submit!'):
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) / 255
    img = np.power(img, gama)
    msrcr_res = automatedMSRCR(img, [15, 80, 200])
    new_img = msrcr_res
    st.write('result of MSRCR:')
    st.image(new_img, channels='BGR')
    st.write('original img:')
    st.image(img, channels='BGR')
