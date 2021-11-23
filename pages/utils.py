import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

def get_dataset(dataset_name):
    data = None

    if dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Upload your dataset(.csv)":
        upload_file = st.file_uploader("Choose a csv file", type ='csv')
        if upload_file is not None:
            data = pd.read_csv(upload_file,header=0,index_col=None)
            st.success("Successfully uploaded")
            target = st.sidebar.selectbox("target",data.columns)
            x = data[data.columns.difference([target])]
            y = data[target]
            feature_names = x.columns
            target_name = target
            target_unique = np.unique(y)
            return x, y, feature_names, target_name, target_unique
        else:
            st.info("Please upload a csv file")
    x = data.data
    y = data.target
    target_name = "target"
    target_unique = data.target_names
    feature_names = data.feature_names
    st.write(f"""
     Your selected dataset is **{dataset_name}**
     """)
    return x, y, feature_names, target_name, target_unique

def add_parameter(model_name):
    params = dict()
    st.sidebar.write("Select parameters")

    if model_name == "SVM":
        c = st.sidebar.slider("C",0.01,15.0, value=1.0)
        degree = 3
        kernel = st.sidebar.selectbox("kernel",("linear","poly","rbf","sigmoid","precomputed"),index=2)
        if kernel == "poly":
            degree = st.sidebar.number_input("degree",step=1, value=3)
        gamma = st.sidebar.radio("gamma",("scale","auto"), index=0)
        params = {'C':c,
                  'kernel':kernel,
                  'degree':degree,
                  'gamma':gamma}

    elif model_name == "KNN":
        n_neighbors = st.sidebar.number_input("n_neighbors",step=1, value=5)
        weights = st.sidebar.radio("weights",("uniform", "distance"), index=0)
        algorithm = st.sidebar.selectbox("algorithm",("auto","ball_tree","kd_tree","brute"),index = 0)
        leaf_size = st.sidebar.number_input("leaf_size",step=1, value=30)
        params = {'n_neighbors':n_neighbors,
                  'weights':weights,
                  'algorithm':algorithm,
                  'leaf_size':leaf_size}

    elif model_name == "Tree":
        criterion = st.sidebar.radio("criterion",("gini","entropy"),index=0)
        splitter = st.sidebar.radio("splitter", ("best","random"), index =0)
        min_samples_split = st.sidebar.number_input("min_samples_split",step=1,value=2)
        min_samples_leaf = st.sidebar.number_input("min_samples_leaf",step=1,value=1)
        params = {'criterion':criterion,
                  'splitter':splitter,
                  'min_samples_split':min_samples_split,
                  'min_samples_leaf':min_samples_leaf}
    return params

def get_model(model_name,params):
    model = None
    if model_name == "SVM":
        model = SVC(C=params['C'],
                    kernel=params['kernel'],
                    degree=params['degree'],
                    gamma=params['gamma'])
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'],
                    weights=params['weights'],
                    algorithm=params['algorithm'],
                    leaf_size=params['leaf_size'])
    elif model_name == "Tree":
        model = DecisionTreeClassifier(criterion=params['criterion'],
                    splitter=params['splitter'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'])
    return model
