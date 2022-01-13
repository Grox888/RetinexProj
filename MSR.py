import numpy as np
import cv2
from retinex import multiScaleRetinex
import streamlit as st
import base64

img = st.file_uploader('imge pls:')
if img != None:
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) / 255
    msr_res = multiScaleRetinex(img, [15, 80, 200])
    new_img = msr_res
    new_img = np.power(10, new_img)
    for i in range(3):
        max = np.max(new_img[:, :, i])
        min = np.min(new_img[:, :, i])
        for h in range(new_img.shape[0]):
            for w in range(new_img.shape[1]):
                new_img[h, w, i] = (new_img[h, w, i] - min) * 255 / (max - min)
    new_img = np.uint8(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.write('result of MSR:')
    st.image(new_img, channels='RGB')
    st.write('original img:')
    st.image(img, channels='RGB')
