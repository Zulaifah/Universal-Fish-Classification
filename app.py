import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Universal Fish Classifier")
st.markdown("### Identify Freshwater Fish Species")

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

@st.cache_resource
def load_models():
    """Load all models"""
    models = {}
    errors = []
    
    # Load ANN-PSO models
    try:
        models['ann'] = joblib.load('ann_pso_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.success("✅ Measurement models loaded")
    except Exception as e:
        errors.append(f"Measurement models: {e}")
        models['ann'] = None
    
    # Load CNN model
    try:
        # Try loading with compile=False to avoid optimizer issues
        models['cnn'] = load_model('universal_fish_cnn_v2.h5', compile=False)
        
        # Load species names
        with open('cnn_class_names.txt', 'r') as f:
            models['cnn_species'] = [line.strip() for line in f.readlines()]
        st.success(f"✅ CNN model loaded ({len(models['cnn_species'])} species)")
    except Exception as e:
        errors.append(f"CNN model: {e}")
        models['cnn'] = None
        models['cnn_species'] = []
    
    if errors:
        st.warning(f"Some models failed to load: {errors[0]}")
    
    return models

models = load_models()

# Create tabs
tab1, tab2 = st.tabs(["📸 Image Classification", "📏 Measurement Classification"])

# ===============================
# TAB 1: IMAGE CLASSIFICATION (CNN)
# ===============================

with tab1:
    st.header("Identify Fish from Photo")
    
    if models['cnn'] is not None:
        st.markdown(f"**Supported Species:** {len(models['cnn_species'])} freshwater fish")
        
        uploaded_file = st.file_uploader("Upload a fish image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Uploaded Fish', width=200)
            
            if st.button("🔍 Identify Species", key="img_predict"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    predictions = models['cnn'].predict(img_array, verbose=0)[0]
                    top_3_idx = np.argsort(predictions)[::-1][:3]
                    
                    with col2:
                        st.success(f"### 🎯 {models['cnn_species'][top_3_idx[0]]}")
                        confidence = predictions[top_3_idx[0]] * 100
                        st.progress(int(confidence))
                        st.caption(f"Confidence: {confidence:.1f}%")
                
                # Show top 3 predictions
                st.subheader("📊 Top 3 Predictions")
                for idx in top_3_idx:
                    st.write(f"- **{models['cnn_species'][idx]}**: {predictions[idx]*100:.1f}%")
    else:
        st.info("📸 CNN model coming soon! Currently being optimized.")

# ===============================
# TAB 2: MEASUREMENT CLASSIFICATION (ANN-PSO)
# ===============================

with tab2:
    st.header("Identify Mugilidae Fish from Measurements")
    
    if models['ann'] is not None:
        st.markdown("Enter the 15 morphometric measurements:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Meristic Features (Counts)")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0, help="First dorsal fin spines+rays")
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0, help="Second dorsal fin spines+rays")
            np_val = st.number_input("NP", value=14.0, step=1.0, help="Pectoral fin rays")
            nc = st.number_input("NC", value=14.0, step=1.0, help="Caudal fin rays")
            nv = st.number_input("NV_Total", value=6.0, step=1.0, help="Ventral fin spines+rays")
            na = st.number_input("NA_Total", value=10.0, step=1.0, help="Anal fin spines+rays")
        
        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0, help="Standard length")
            pl = st.number_input("PL", value=35.0, step=5.0, help="Pectoral fin length")
            bh = st.number_input("BH", value=40.0, step=5.0, help="Body height")
            hl = st.number_input("HL", value=35.0, step=5.0, help="Head length")
            
            st.subheader("Truss Features (Sum in mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0)
        
        if st.button("🔍 Identify Species", key="meas_predict", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                  head, ant, mid, post, tail]])
            scaled = models['scaler'].transform(features)
            prediction = models['ann'].predict(scaled)[0]
            species = models['label_encoder'].inverse_transform([prediction])[0]
            
            st.success(f"### 🎯 Predicted Species: **{species}**")
            
            # Show probabilities
            probabilities = models['ann'].predict_proba(scaled)[0]
            prob_df = pd.DataFrame({
                'Species': models['label_encoder'].classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.subheader("📊 Species Probabilities")
            st.bar_chart(prob_df.set_index('Species'))
    else:
        st.error("Measurement models not loaded. Please check files.")

st.markdown("---")
st.caption("Powered by ANN-PSO + CNN | FYP Project")
