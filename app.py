import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Classifier")
st.markdown("### Identify Mullet Species Using ANN-PSO")

# Load only the measurement models
@st.cache_resource
def load_models():
    try:
        ann = joblib.load('ann_pso_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return ann, scaler, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

ann, scaler, le = load_models()

if ann:
    st.success("✅ ANN-PSO Model Loaded Successfully!")
    
    st.markdown("""
    ### Enter the 15 morphometric measurements:
    
    **Meristic Features (Counts):**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        nd1 = st.number_input("ND1_Total (First Dorsal Fin)", value=4.0, step=1.0)
        nd2 = st.number_input("ND2_Total (Second Dorsal Fin)", value=6.0, step=1.0)
        np_val = st.number_input("NP (Pectoral Fin Rays)", value=14.0, step=1.0)
        nc = st.number_input("NC (Caudal Fin Rays)", value=14.0, step=1.0)
        nv = st.number_input("NV_Total (Ventral Fin)", value=6.0, step=1.0)
        na = st.number_input("NA_Total (Anal Fin)", value=10.0, step=1.0)
    
    with col2:
        sl = st.number_input("SL (Standard Length mm)", value=150.0, step=10.0)
        pl = st.number_input("PL (Pectoral Length mm)", value=35.0, step=5.0)
        bh = st.number_input("BH (Body Height mm)", value=40.0, step=5.0)
        hl = st.number_input("HL (Head Length mm)", value=35.0, step=5.0)
        
        st.markdown("**Truss Features (Sum in mm):**")
        head = st.number_input("Head_Truss", value=80.0, step=10.0)
        ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
        mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
        post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
        tail = st.number_input("Tail_Truss", value=200.0, step=20.0)
    
    if st.button("🔍 Identify Species", type="primary"):
        features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                              head, ant, mid, post, tail]])
        scaled = scaler.transform(features)
        prediction = ann.predict(scaled)[0]
        species = le.inverse_transform([prediction])[0]
        
        st.success(f"### 🎯 Predicted Species: **{species}**")
        
        # Show probability distribution
        probabilities = ann.predict_proba(scaled)[0]
        prob_df = pd.DataFrame({
            'Species': le.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.subheader("📊 Species Probabilities")
        st.bar_chart(prob_df.set_index('Species'))
    
    st.info("📌 Note: Image classification coming soon! Currently supporting 5 Mugilidae species.")
    
else:
    st.warning("""
    ### Models not loaded
    
    Please ensure these files are in your repository:
    - `ann_pso_model.pkl`
    - `scaler.pkl`  
    - `label_encoder.pkl`
    """)

st.markdown("---")
st.caption("Powered by ANN-PSO Optimization | FYP Project")
