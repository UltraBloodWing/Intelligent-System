#make a gui

# 3 components: save time and date, takes audio input, converts it, and predicts the correct word and stores
#in a file, download the file

#references for the code
#https://www.programiz.com/python-programming/datetime/current-datetime#:~:text=today()%20method%20to%20get,representing%20date%20in%20different%20formats.
#https://github.com/stefanrmmr/streamlit_audio_recorder/blob/main/streamlit_app.py
#https://towardsdatascience.com/streamlit-hands-on-from-zero-to-your-first-awesome-web-app-2c28f9f4e214

#Required imports 
import pandas as pd
import numpy as np
import cv2
from datetime import date
import os
from keras.models import load_model
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import streamlit.components.v1 as components
import librosa
import matplotlib.pyplot as plt
import librosa.display

#The Manual Codes

audio = "streamlit_audio_25_05_202219_45_00.wav"

hop_length = 512
window_size = 1024
window = np.hanning(window_size)

#Convert image to spectrogram
y,sr=librosa.load(audio) #load the file
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128)
spectrogram = librosa.power_to_db(spectrogram)
spectrogram = spectrogram.astype(np.float32)

#Save the spectrogram image
out  = librosa.core.spectrum.stft(y, n_fft = window_size, hop_length = hop_length, window=window)
out = 2 * np.abs(out) / np.sum(window)

fig = plt.Figure()
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')

image = "something.png"

fig.savefig(image)

#Load the model
saved_model = load_model("model2.h5")

#Resize the image for model prediction  
img = cv2.imread(image)
resized_img = cv2.resize(img, (331, 331)).reshape(-1, 331, 331, 3)

#Model prediction
print(saved_model.predict(resized_img))

#GUI
#Loading the component from REACT responsible for recording
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
#Initialise audio rec object functionaly
st_audiorec = components.declare_component("st_audiorec", path=build_dir)

#Add wordings to the GUI
st.title("Smart Meeting Minutes")
st.markdown("An Audio to text conversion tool")
st.markdown(date.today())
st.write('\n\n')

#Streamlit audio of userinput
st_audiorec()

