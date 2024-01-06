import pickle
import numpy as np

x_train = pickle.load(open("x_train.pkl", "rb"))
x_train_norm = pickle.load(open("x_train_norm.pkl", "rb"))
y_train = pickle.load(open("y_train.pkl", "rb"))


# model import
from Number_classify import classifiy_number

import streamlit as st
from random import randint

from matplotlib import pyplot as plt

def predicted_values():
    result = ""
    for i in st.session_state.rand_values:
        pred = classifiy_number(x_train_norm[i])
        result += str(pred)
    return result

def original_values():
    result = ""
    for i in st.session_state.rand_values:
        result += str(y_train[i])
    return result

def set_number_images(n_cols):
    random_values = []
    for col in st.columns(spec=n_cols):
        index = randint(0, len(x_train))
        random_values.append(index)
        with col:
            fig, ax = plt.subplots()
            ax.axis('off')
            plt.imshow(x_train[index])
            st.pyplot(fig)
    
    st.session_state.rand_values = random_values

st.title("Robut")

st.text("Select any count to get Random Values")

n_cols = st.select_slider(label="Select Count", options=[i for i in range(1,16)])
refresh = st.button(label="Refresh")
if n_cols:
    n_cols = int(n_cols)
    set_number_images(n_cols)

st.header("Predictions ")

o = original_values()
p = predicted_values()

st.text(f"Original Value : {o}")
st.text(f"Model predicted Value : {p}")

count = 0
for i in range(len(o)):
    if o[i] == p[i]:
        count += 1

st.text(f"Matching Percentage : {(count / len(o)) * 100}")
