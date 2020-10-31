import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def Standardscaler(features):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X