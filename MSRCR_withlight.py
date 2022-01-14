import numpy as np
import cv2
from retinex import automatedMSRCR
import streamlit as st
import base64

img = st.file_uploader('imge pls:')
I = st.slider('光强I', 0.0, 1.0, 0.3, 0.01)
sigma = st.slider('光照范围sigma', 15, 500, 200, 5)
if st.button('submit!'):
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) / 255
    h = img.shape[0] // 4
    w = img.shape[1] // 4

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j, :] = img[i, j, :] + I * np.exp(-(i ** 2 + j ** 2) / (sigma ** 2))
    img = np.clip(img, 0, 1)
    msrcr_res = automatedMSRCR(img, [15, 80, 200])
    new_img = msrcr_res
    st.write('result of MSRCR:')
    st.image(new_img, channels='BGR')
    st.write('original img:')
    st.image(img, channels='BGR')