import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache(suppress_st_warning=True)
def SOM(X_train, X_test, y_train, y_test, k, epoch, neigh_fun):
    som = MiniSom(x=12, y=12, input_len=k, sigma=1.0, learning_rate=0.5, neighborhood_function=neigh_fun,)
    som.pca_weights_init(X_train)
    st.write('Weights before training data')
    st.write(som.get_weights())
    som.train_random(X_train, int(epoch))
    st.write('Weights After training data')
    st.write(som.get_weights())
    st.write("Accuracy : ", accuracy_score(y_test, classify(som, X_test, X_train, y_train)))


def classify(som, data, X_train, y_train):
  
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result
