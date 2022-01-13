import numpy as np
import cv2
from retinex import automatedMSRCR
import streamlit as st
import base64

img = st.file_uploader('imge pls:')
if img != None:
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) / 255
    msrcr_res = automatedMSRCR(img, [15, 80, 200])
    new_img = msrcr_res
    st.write('result of MSRCR:')
    st.image(new_img, channels='RGB')
    st.write('original img:')
    st.image(img, channels='RGB')