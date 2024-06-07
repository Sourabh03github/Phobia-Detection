import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import streamlit as st
from model_training_phase import train_model
from model_loader import load_model

import pandas as pd
import streamlit as st

# Load the trained emotion classification model
model = load_model()  # Assuming you have a model_loader.py function to load the model

# Streamlit app configuration
st.title("Please Connect The EEG Electrodes")

st.write("""
Before proceeding, here are some tips for managing your emotions:

1. **Active Listening:** Focus on genuinely listening to your feelings.
2. **Calm Environment:** Find a quiet, comfortable space.
3. **Breathing Exercises:** Practice deep breaths to reduce anxiety and stress.
4. **Reassurance:** Validate your feelings and acknowledge that it's okay to feel upset.
5. **Physical Comfort:** If comfortable, consider gentle touch for reassurance.
""")

video_file = '/Users/avishkarborkar/Desktop/FY Project/videoplayback.mp4' # Replace with your video URL
#video_play_button = st.button("Play Video")

st.video(video_file)


st.write("Upload a CSV file for batch prediction:")

# File uploader for CSV input
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
# Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    prediction = model.predict(data)
    st.write("Original Data:")
    st.dataframe(data.head())
    emotion = prediction[0]
    emotion_map = {0: 'NEGATIVELY', 1: 'POSITIVELY', 2: 'NEUTRAL'}  
    predicted_emotion = emotion_map[emotion]
    st.write(f"The patient reacted {predicted_emotion} to the simulation.")
    



