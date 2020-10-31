import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def train_split_data(features, target):
    st.subheader('Training and Testing Data')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
    st.write('Training features', X_train.shape)
    st.write('Testing features', X_test.shape)
    st.write('Training label', y_train.shape)
    st.write('Testing label', y_test.shape)
    return X_train, X_test, y_train, y_test