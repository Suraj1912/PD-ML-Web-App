import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from features_selection  import selectkBest
from standardization import Standardscaler
from split_data import train_split_data
from mlp import MLP
from svm import SVM
from som import SOM
from lvq import LVQ


def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)

    activities = ['Explore Data', 'EDA', 'Model Selection']

    choice = st.sidebar.selectbox('Select Activity', activities)

    st.title('Parkinson Disease Classification')

    dataset = st.file_uploader('Upload Dataset', type= 'csv')
    if dataset is not None:
        dataset = pd.read_csv(dataset)

       

        if choice == 'Explore Data':
            exploreData(dataset)

        elif choice == 'EDA':
            eda(dataset)


        elif choice == 'Model Selection':
            features, target = divide(dataset)
            if st.sidebar.checkbox('Features Selection'):
            # select = st.sidebar.radio('Select Methods', ('SelectKBest', 'Standardization'))
            # if select == 'SelectKBest':
                st.write('Features Selection using SelectKBest')
                k = st.sidebar.slider('Set number of top features to select', min_value=5, max_value=50)
                st.sidebar.write('you selected', k)
                features = selectkBest(k, features, target)

            if st.sidebar.checkbox('Standardization'):
            # if select == 'Standardization':
                st.write('Standardization')
                st.write(features.shape)
                features = Standardscaler(features)
                st.write(features)

            X_train, X_test, y_train, y_test = train_split_data(features, target)


            # st.subheader('Model Building')
            model = st.sidebar.selectbox('Select ML Algorithms', ['MultiLayer Perceptron', 'Support Vector Machine', 'Self Organizing Maps', 'Learning Vector Quantization'])            
            if model == 'MultiLayer Perceptron':
                st.subheader('MultiLayer Perceptron')
                st.sidebar.subheader("Model Hyperparameters")
                hidden_layers = st.sidebar.slider('Hidden Layers', min_value=100, max_value=500, step=100)
                max_iter = st.sidebar.slider('No. of Iterations', min_value=100, max_value=1000, step=100)
                activation_func = st.sidebar.selectbox('Select Activation Function', ['identity', 'logistic', 'relu', 'tanh'])
                solver = st.sidebar.selectbox('Select Solver', ['adam', 'sgd'])
                if st.sidebar.button('Apply', 'mlp'):
                    MLP(X_train, X_test, y_train, y_test, hidden_layers, activation_func, solver, max_iter)

            elif model == 'Support Vector Machine':
                st.subheader('Support Vector Machine')
                st.sidebar.subheader("Model Hyperparameters")
                start = st.sidebar.number_input('From', min_value=1.0, max_value=1000.0, step=1.0)
                end = st.sidebar.number_input('To', min_value=1.0, max_value=1000.0, step=1.0)
                C = np.arange(start, end+1)
                kernel = st.sidebar.multiselect('Select Kernels', ['linear', 'poly', 'rbf', 'sigmoid'])
                if st.sidebar.button('Apply', 'svm'):
                    SVM(X_train, X_test, y_train, y_test, C, kernel)

            elif model == 'Self Organizing Maps':
                st.subheader('Self Organizing Maps')
                st.sidebar.subheader("Model Hyperparameters")
                epoch = st.sidebar.slider('Set epoch', min_value=50.0, max_value=1500.0, step=50.0)
                neighbor_fun = st.sidebar.selectbox('Select Neighborhood Function', ['gaussian', 'triangle'])
                if st.sidebar.button('Apply', 'som'):
                    SOM(X_train, X_test, y_train, y_test, k, epoch, neighbor_fun)

            else:
                st.subheader('Learning Vector Quantization')
                epoch = st.sidebar.slider('Set epoch', min_value=50.0, max_value=1500.0, step=50.0)
                learn_rate = st.sidebar.number_input('Set Learning Rate', min_value=0.1, max_value=1.1, step=0.1)
                if st.sidebar.button('Apply', 'lvq'):
                    LVQ(X_train, X_test, y_train, y_test, epoch, learn_rate)

    else:
        st.write('Upload dataset first!!!')


def exploreData(dataset):
    st.subheader('Explore Dataset')
    size = st.slider('Select size to show', 1, dataset.shape[0])
    st.dataframe(dataset.head(size))

    if st.checkbox('Data Shape'):
        st.write(dataset.shape)

    if st.checkbox('Show Columns'):
        st.write(dataset.columns.to_list())

    if st.checkbox('Data Summary'):
        st.write(dataset.describe())

    if st.checkbox('Check for null values'):
        st.write(dataset.isnull().sum())
        # if dataset.isnull().any() == 0:
        #     st.write('There are no null values')


def eda(dataset):
    st.subheader('Exploratory Data Analysis')

    if st.sidebar.checkbox('Correlation Matrix'):
        radio = st.sidebar.radio('Select random columns OR give ranges', ('Selct random columns', 'Give ranges'))
        try:
            if radio == 'Selct random columns':
                selected_cols = st.multiselect('Select Columns', dataset.columns.to_list())
                st.write(sns.heatmap(dataset[selected_cols].corr(), annot=True))
            else:
                start = int(st.number_input('Starting index', step=1.0))
                end = int(st.number_input('Ending index', step=1.0))
                st.write('Start : ', start)
                st.write('End : ', end)
                st.write(sns.heatmap(dataset[dataset.columns[start:end]].corr(), annot=True))
            st.pyplot()
        except:
            st.write('select columns or ranges')

        
       
    if st.sidebar.checkbox('Univariate Analysis'):
        plots = st.sidebar.selectbox('Select Plots', ['Countplot', 'Distplot', 'Boxplot'])
        if plots == 'Countplot':
            uni_cols = st.selectbox('Select Column', dataset.columns.to_list())
            if st.checkbox('hue'):
                st.write(sns.countplot(dataset[uni_cols], hue=dataset['class']))
            else:
                st.write(sns.countplot(dataset[uni_cols]))

        if plots == 'Distplot':
            uni_cols = st.selectbox('Select Column', dataset.columns.to_list())
            st.write(sns.distplot(dataset[uni_cols]))

        if plots == 'Boxplot':
            uni_cols = st.selectbox('Select Column', dataset.columns.to_list())
            st.write(sns.boxplot(dataset[uni_cols]))
        st.pyplot()

    if st.sidebar.checkbox('Bivariate Analysis'):
        bi_plots = st.sidebar.selectbox('Select Plots', ['Boxplot', 'Scatterplot'])
        if bi_plots == 'Boxplot':
            bi_cols = st.selectbox('Select Columns', dataset.columns.to_list())
            st.write(sns.boxplot(y = dataset[bi_cols], x = dataset['class']))
            st.pyplot()

        if bi_plots == 'Scatterplot':
            x = st.selectbox('Select for x axis', dataset.columns.to_list())
            y = st.selectbox('Select for y axis', dataset.columns.to_list())
            st.write(sns.scatterplot(x=x, y=y, hue='class', data=dataset))
            st.pyplot()
    

def divide(dataset):
    features = dataset.drop(['id', 'class'], axis=1)
    target = dataset['class']

    return features, target



if __name__ == '__main__':
    main()