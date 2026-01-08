import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Mark Predictor',page_icon='✍️',layout='centered')

st.title('Student Marks Predictor')
st.write('Enter The number of Hours Studied **1 To 10** and **CLICK PREDICT** to see predicted Marks')

# Load the model

def load_model(model):
    with open(model,'rb')as f:
        slr = pickle.load(f)
    return slr

try:
    model = load_model('slr.pkl')
except Exception as e:
    st.error('Your pickle file not found.....')
    st.exception('Failed to load the Model:',e)
    st.stop()


hours = st.number_input('Hours_Studied',
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.1,
                        format='%.1f')

if st.button('Predict'):
    try:
        X = np.array([[hours]])
        predictions = model.predict(X)
        predictions = predictions[0]

        st.success(f'Predicted Marks: {predictions:.1f}')
        st.write("Note: This is ML Model Prediction **Result May Vary**")
    except Exception as e:
        st.error(f'Prediction failed: {e}')

