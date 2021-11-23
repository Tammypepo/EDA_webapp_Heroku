import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

def app():
    st.write("""
        ### Model Building
        """)
    dataset_name = st.sidebar.selectbox("Select Dataset",("Breast Cancer","Iris","Wine","Upload your dataset(.csv)"),index=1)
    try:
        x,y,feature_names,target_name,target_unique = utils.get_dataset(dataset_name)
        df = pd.DataFrame(x,columns = feature_names)

        model_name = st.sidebar.selectbox("Select classifier",("SVM","KNN","Tree"))
        st.write(f"""
         Your selected classifier is **{model_name }**
         """)

        selected_columns = st.multiselect("Select preferred columns", df.columns, default= list(df.columns))
        df = df[selected_columns]

        all_columns = df.columns.to_list()
        st.dataframe(df,800,300)



        params = utils.add_parameter(model_name)

        if st.sidebar.button("build model"):

            model = utils.get_model(model_name,params)

            x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=0)

            model.fit(x_train,y_train)

            y_pred = model.predict(x_test)

            score =classification_report(y_test,y_pred,output_dict=True)
            df_score = pd.DataFrame(score).transpose()

            st.write("""### Your model performance""")
            st.dataframe(df_score.iloc[:-3])
            st.write("""#### Your accuracy """,round(df_score.iloc[-3,0]*100,2),"%")
            st.write("""Your test set prediction """)
            x_test['target'] = y_test
            x_test['predicted'] = y_pred
            x_test['result'] = y_test == y_pred
            st.dataframe(x_test)
        else:
            pass
    except:
        pass
