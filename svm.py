import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache(suppress_st_warning=True)
def SVM(X_train, X_test, y_train, y_test, C, kernel):
    accuracy_linear = []
    accuracy_poly = []
    accuracy_rbf = []
    accuracy_sig = []

    st.write("Mean accuracy of Kernals : ")
    plt.grid()

    if 'linear' in kernel:
        for c in C:
            SVM = SVC(kernel='linear', C=c, random_state=1)
            SVM.fit(X_train, y_train)
            y_pred = SVM.predict(X_test)
            accuracy_linear.append(accuracy_score(y_test, y_pred)*100)

        plt.plot(C, accuracy_linear, label="Linear")
        st.write("Linear : ", np.mean(accuracy_linear))

    if 'poly' in kernel:
        for c in C:
            SVM = SVC(kernel='poly', C=c, random_state=1)
            SVM.fit(X_train, y_train)
            y_pred = SVM.predict(X_test)
            accuracy_poly.append(accuracy_score(y_test, y_pred)*100)

        plt.plot(C, accuracy_poly, label="Poly")
        st.write("Poly : ", np.mean(accuracy_poly))

    if 'rbf' in kernel:
        for c in C:
            SVM = SVC(kernel='rbf', C=c, random_state=1)
            SVM.fit(X_train, y_train)
            y_pred = SVM.predict(X_test)
            accuracy_rbf.append(accuracy_score(y_test, y_pred)*100)

        plt.plot(C, accuracy_rbf, label="RBF")
        st.write("RBF : ", np.mean(accuracy_rbf))

    if 'sigmoid' in kernel:
        for c in C:
            SVM = SVC(kernel='sigmoid', C=c, random_state=1)
            SVM.fit(X_train, y_train)
            y_pred = SVM.predict(X_test)
            accuracy_sig.append(accuracy_score(y_test, y_pred)*100)

        plt.plot(C, accuracy_sig, label="Sigmoid")
        st.write("Sigmoid : ", np.mean(accuracy_sig))

    
    plt.xlabel("Value of C")
    plt.ylabel("Accuracy in %")
    plt.title("Support Vector Classifier")
    plt.legend()
    st.pyplot()

    # grid = GridSearchCV(estimator=SVC(), param_grid={'C' : C, 'kernel' : kernel} )
    # grid.fit(X_train, y_train)
    # st.write(grid.best_estimator_)


    
    
    
    
    
    