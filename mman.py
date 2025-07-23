import streamlit as st
import PIL 
from PIL import Image
import matplotlib.pyplot  as plt
import numpy as np
import pandas as pd

import joblib

model = joblib.load(r"D:\New folder\shoesgit\model.kathford.pt")
def image_process(img):
    img_pil = Image.open(img)
    img_ary = np.array(img_pil)
    img_flat = img_ary.flatten()
    df_t = pd.DataFrame(img_flat).T
 
    p = model.predict(df_t)
    return p
    
st.title("Shoes Classification ,Machine Learning")
file =st.file_uploader("Upload your file ,type ['jpg','jpeg']")
try:
    if file is not None:
        i = Image.open(file)
        st.write(i)
        pr= image_process(file)
        st.write(f"The predict brand is {pr}")
    else:
        st.write("Empty file")
except Exception as e:
      st.error(f"An error occurred: {e}")