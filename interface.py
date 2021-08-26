import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

model_1 = tf.keras.models.load_model("my_model_0-51.hdf5")
model_2 = tf.keras.models.load_model("C:\stage_data_inception\code\inception.hdf5")
model_3 = tf.keras.models.load_model("C:\stage_data_inception\code\my_model_VGG1.hdf5")

st.set_page_config(
    page_title="DeepDetect",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.write("""
         #            DeepDetect
         ####         Votre solution pour detecter les documents falsifiés   
         """
         )
st.sidebar.image("C:/stage_data_inception/datainception-logo.png", use_column_width=True)

choix_model=st.sidebar.selectbox('Veuillez choisir le modele à utiliser ',('modele interne','model_inception','model_VGG16'))


file = st.sidebar.file_uploader("Veuillez importer une image", type=["jpg", "png"])

        
        

def import_and_predict(image, model):
    
        size = (224,224)    
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, size, interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)
        
        return prediction


if file is not None:

    image = Image.open(file)
    st.image(image, use_column_width=True)
    if (choix_model=='modele interne'):
        prediction = import_and_predict(image, model_1)
    if (choix_model=='model_inception'):
        prediction = import_and_predict(image, model_2)    
    if (choix_model=='model_VGG16'):
        prediction = import_and_predict(image, model_3)
    
    if (st.sidebar.button('vérifier le passeport')):
         
        if np.argmax(prediction) == 0:
            st.write("""
            ##  c'est un passeport falsifié
             """)
        else:
            st.write("""
            ##   c'est un passeport vrai
             """)
    
        df = pd.DataFrame(prediction, columns = ['falsifié','réel'])
        st.write(df)





