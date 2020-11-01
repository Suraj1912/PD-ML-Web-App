# **Parkinson’s Disease Classification**

View my work from [here](https://pd-ml-web-app.herokuapp.com/).

## About Parkinson Disease →

Parkinson's disease is a brain disorder that leads to shaking, stiffness, and difficulty
with walking, balance, and coordination. Parkinson's symptoms usually begin
gradually and get worse over time. As the disease progresses, people may have
difficulty walking and talking. Parkinson's disease dementia becomes common in
the advanced stages of the disease. Depression and anxiety are also common,
occurring in more than a third of people with PD. Other symptoms include
sensory, sleep, and emotional problems. The main motor symptoms are
collectively called "parkinsonism", or a "parkinsonian syndrome". 


## About Parkinson Disease Dataset →

Dataset is taken from kaggle [PD dataset](https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features). In this study, we aim to analyze and diagnose patients with Parkinson Disease (PD) on speech datasets. The data used in this study were gathered from 188 patients
with PD (107 men and 81 women) with ages ranging from 33 to 87 (65.1Â±10.9)
at the Department of Neurology Faculty of Medicine, Istanbul University. The
control group consists of 64 healthy individuals (23 men and 41 women) with ages
varying between 41 and 82 (61.1Â±8.9). During the data collection process, the
microphone is set to 44.1 KHz and following the examination, the sustained
phonation of the vowel /a/ was collected from each subject with three repetitions.


## Attribute Information→

Various speech signal processing algorithms including Time Frequency Features,
Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform based
Features, Vocal Fold Features and TWQT features have been applied to the speech
recordings of Parkinson's disease (PD) patients to extract clinically useful
information for PD assessment. There are 754 features applied on 756 peoples to
tell whether it is Parkinson patient or not.


## Steps to Classify →

I was perform different tasks to classify the Parkinson patients.

1. Load dataset
Loaded Parkinson disease speech dataset from Kaggle

2. Exploring dataset → check any null values, shape, columns, summary
Checked dataset shape having 756 rows and 755 columns. There are no null
values in dataset. Also having one class column with 0 and 1 values that’s
basically our label.

3. Exploratory Data Analysis → draw different plots univariate, bivariate
This step plays an important role in making classification model. In dataset
id column is numeric, gender is categorical in nature while others have
columns have continuous values. To check for strong relation in columns I
used correlation matrix. Different plots such as countplot, scatterplot,
distplot, boxplot is used for detecting any outliers, distribution in columns.

4. Feature Selection
From above step I know all about the data so, now I can easily drop those
columns which are not much responsible to classify. For that I used
SelectKBest module from sklearn library in python. This gives me the
features with the scores which are needed for the labels.

5. Standardization
To keep data on same scale we require to standardize the data. By doing this
our model can efficiently do better job and give higher accuracy for same.

6. Modelling
Now our data is ready and waiting to impose on model to do classification.
In this I have used Multilayer Perceptron, Support Vector Machine, SelfOrganizing Maps and Learning Vector Quantization.

## Libraries

Python libraries are used
```
1. numpy  # For arrays manipulation
2. Pandas  # For data manipulation
3. Matplotlib # For plots
4. Seaborn  # For plots
5. Scikit-learn  # For python pre-defined libraries
5. Minisom   # For SOM implementation
6. Streamlit  # For front end model delivery
```

## Requirements

To run the software once need to install required libraries. For that simply run command in cmd
```
pip install -r requirements.txt
```

## Run on Local Machine

Clone repo from here and navigate to directory streamlit Web App then run command 
```
streamlit run app.py
```

## Specifications

1. Windows 8 and Above
2. Python3.6 or later 64-bit
