import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache(suppress_st_warning=True)
def MLP(X_train, X_test, y_train, y_test, hidden_layer, activ, solver, max_iter):
    perceptron_model = MLPClassifier(hidden_layer_sizes=(hidden_layer,), early_stopping=False, activation=activ, solver=solver, max_iter=max_iter)
    perceptron_model.fit(X_train, y_train)
    st.success('Model Trained Successfully')
    y_pred = perceptron_model.predict(X_test)
    st.success('Model Tested Successfully')
    unique, counts = np.unique(y_pred, return_counts=True)
    res = dict(zip(unique, counts))
    st.write('0 : ', res[0], '1 : ', res[1])
    plt.plot(perceptron_model.loss_curve_)
    plt.legend(['Loss'])
    plt.xlabel('n iterations')
    plt.title('Loss each iteration')
    st.pyplot()
    st.write('updated Weights from input layer to hidden layer')
    st.write(perceptron_model.coefs_[0])
    st.write('updated Weights from hidden layer to output layer')
    st.write(perceptron_model.coefs_[1])

    metrices(y_test, y_pred)

@st.cache(suppress_st_warning=True)
def metrices(y_test, y_pred):
    st.write('Accuracy : ', accuracy_score(y_pred, y_test))
    st.write('Confusion matrix : ', sns.heatmap(confusion_matrix(y_pred, y_test), annot=True))
    st.pyplot()