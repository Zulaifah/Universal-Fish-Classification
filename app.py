
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Universal Fish Classifier")
st.markdown("### Identify Freshwater Fish Species")

# Set memory growth to avoid issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

@st.cache_resource
def load_models():
    try:
        # Try loading with different methods
        cnn = load_model('universal_fish_cnn.h5', compile=False)
        ann = joblib.load('ann_pso_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        
        with open('cnn_class_names.txt', 'r') as f:
            species = [line.strip() for line in f.readlines()]
        
        return cnn, ann, scaler, le, species
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

cnn_model, ann_pso, scaler, label_encoder, cnn_species = load_models()

if cnn_model is not None:
    st.success(f"✅ Models loaded! CNN ready for {len(cnn_species)} species")
    
    tab1, tab2 = st.tabs(["📸 Image Classification", "📏 Measurement Classification"])
    
    with tab1:
        st.header("Identify Fish from Photo")
        uploaded = st.file_uploader("Upload a fish image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, width=200)
            
            if st.button("🔍 Identify Species", key="img_predict"):
                with st.spinner("Analyzing image..."):
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = cnn_model.predict(img_array, verbose=0)[0]
                    top_idx = np.argmax(predictions)
                    
                    with col2:
                        st.success(f"### 🎯 {cnn_species[top_idx]}")
                        confidence = predictions[top_idx] * 100
                        st.progress(int(confidence))
                        st.caption(f"Confidence: {confidence:.1f}%")
                    
                    # Show top 3
                    top_3 = np.argsort(predictions)[::-1][:3]
                    st.subheader("Top 3 Predictions")
                    for idx in top_3:
                        st.write(f"- {cnn_species[idx]}: {predictions[idx]*100:.1f}%")
    
    with tab2:
        st.header("Identify Mugilidae from Measurements")
        st.info("Enter the 15 morphometric measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Meristic Features**")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)
        
        with col2:
            st.markdown("**Morphometric Features (mm)**")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=35.0, step=5.0)
            bh = st.number_input("BH", value=40.0, step=5.0)
            hl = st.number_input("HL", value=35.0, step=5.0)
            
            st.markdown("**Truss Features (mm)**")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0)
        
        if st.button("🔍 Identify Species", key="meas_predict"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                  head, ant, mid, post, tail]])
            scaled = scaler.transform(features)
            pred = ann_pso.predict(scaled)[0]
            species = label_encoder.inverse_transform([pred])[0]
            st.success(f"### 🎯 Predicted Species: **{species}**")
else:
    st.warning("""
    ## Models not loaded
    
    Please ensure these files are uploaded:
    - `universal_fish_cnn.h5` (CNN model)
    - `ann_pso_model.pkl` (ANN-PSO model)
    - `scaler.pkl`
    - `label_encoder.pkl`
    - `cnn_class_names.txt`
    
    If files are uploaded, there might be a version mismatch.
    """)

st.markdown("---")
st.caption("Powered by ANN-PSO + CNN | FYP Project")
