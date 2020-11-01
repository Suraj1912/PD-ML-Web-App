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

    activities = ['Description', 'Explore Data', 'EDA', 'Model Selection']

    choice = st.sidebar.selectbox('Select Activity', activities)

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Parkinson Disease Classification</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""<br><span>Made by </span><span style='color: #FF0000;'>Suraj Patil</span>""", unsafe_allow_html=True)

    # dataset = st.file_uploader('Upload Dataset', type= 'csv')
    # if dataset is not None:
    dataset = pd.read_csv('pd_speech_features.csv')

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

    # else:
    #     st.write('Upload dataset first!!!')

    else:
        st.markdown(describe(), unsafe_allow_html=True)


def exploreData(dataset):
    st.subheader('Explore Dataset')
    size = st.slider('Select size to show', 1, 20)
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


def describe():
    para = """
    <h2>About Parkinson Disease</h2>
    <p>Parkinson's disease is a brain disorder that leads to shaking, stiffness, and difficulty with walking, 
    balance, and coordination. Parkinson's symptoms usually begin gradually and get worse over time. As the disease 
    progresses, people may have difficulty walking and talking. Parkinson's disease dementia becomes common in the 
    advanced stages of the disease. Depression and anxiety are also common, occurring in more than a third of people 
    with PD. Other symptoms include sensory, sleep, and emotional problems. The main motor symptoms are collectively 
    called "parkinsonism", or a "parkinsonian syndrome".</p>
    <h2>About Parkinson Disease Dataset </h2>
    <p>Dataset is taken from kaggle <a href='https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features'>PD dataset</a>. 
    In this study, we aim to analyze and diagnose patients with Parkinson 
    Disease (PD) on speech datasets. The data used in this study were gathered from 188 patients with PD (107 men and 81 women) 
    with ages ranging from 33 to 87 (65.1Â±10.9) at the Department of Neurology Faculty of Medicine, Istanbul University. 
    The control group consists of 64 healthy individuals (23 men and 41 women) with ages varying between 41 and 82 (61.1Â±8.9). 
    During the data collection process, the microphone is set to 44.1 KHz and following the examination, the sustained phonation 
    of the vowel /a/ was collected from each subject with three repetitions.</p>
    <h2>Attribute Information</h2>
    <p>Various speech signal processing algorithms including Time Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), 
    Wavelet Transform based Features, Vocal Fold Features and TWQT features have been applied to the speech recordings of Parkinson's 
    disease (PD) patients to extract clinically useful information for PD assessment. There are 754 features applied on 756 peoples to 
    tell whether it is Parkinson patient or not.<br>1. Numeric --> id column <br>2. Categorical --> gender and class <br>3. Continuous --> 752 columns</p>
    <h2>Steps to Classify</h2>
    <p>I was perform different tasks to classify the Parkinson patients.

1. <b>Load dataset</b> Loaded Parkinson disease speech dataset from Kaggle

2. <b>Exploring dataset</b> → check any null values, shape, columns, summary Checked dataset shape having 756 rows and 755 columns. There are no null values in dataset. Also having one class column with 0 and 1 values that’s basically our label.

3. <b>Exploratory Data Analysis</b> → draw different plots univariate, bivariate This step plays an important role in making classification model. In dataset id column is numeric, gender is categorical in nature while others have columns have continuous values. To check for strong relation in columns I used correlation matrix. Different plots such as countplot, scatterplot, distplot, boxplot is used for detecting any outliers, distribution in columns.

4. <b>Feature Selection</b> From above step I know all about the data so, now I can easily drop those columns which are not much responsible to classify. For that I used SelectKBest module from sklearn library in python. This gives me the features with the scores which are needed for the labels.

5. <b>Standardization</b> To keep data on same scale we require to standardize the data. By doing this our model can efficiently do better job and give higher accuracy for same.

6. <b>Modelling</b> Now our data is ready and waiting to impose on model to do classification. In this I have used Multilayer Perceptron, Support Vector Machine, SelfOrganizing Maps and Learning Vector Quantization.</p>
    """

    return para



if __name__ == '__main__':
    main()