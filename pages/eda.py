import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pages import utils
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def app():
    cmap = LinearSegmentedColormap.from_list(
        name='test',
        colors=['red','pink','white','#98FF98','#32CD32']
    )
    #set subtitle
    st.write("""
        ### Let's Explore different datasets
        """)
    dataset_name = st.sidebar.selectbox("Select Dataset",("Breast Cancer","Iris","Wine","Upload your dataset(.csv)"),index=1)

    try:
        x,y,feature_names,target_name,target_unique = utils.get_dataset(dataset_name)
        df = pd.DataFrame(x,columns = feature_names)

        selected_columns = st.multiselect("Select preferred columns", df.columns, default= list(df.columns))
        df_show = df[selected_columns]
        df_plot = df_show.copy()
        df_plot[target_name] = y
        all_columns = df_plot.columns.to_list()
        st.dataframe(df_show,800,300)
        st.sidebar.write("Please select EDA options")

        if st.sidebar.checkbox("Shape",value=True):
            st.write(f"Shape of data(not included target) is :", df_show.shape)

        if st.sidebar.checkbox("Unique target amount",value=True):
            st.write(f"Unique target name is :", str(target_unique))
            st.write(f"Unique target amount is :", len(np.unique(y)))

        if st.sidebar.checkbox("Data types & null value",value=True):

            dtypes = df_plot.dtypes.apply(lambda x: x.name).to_dict()
            df_dtypes  =pd.DataFrame.from_dict(dtypes, orient='index',columns=["type"])
            st.markdown("""---""")

            col1, col2  = st.columns(2)
            with col1:
                st.write("""**data type**""")
                st.dataframe(df_dtypes)
            with col2:
                st.write("""**null values**""")
                st.dataframe(df_plot.isna().sum().rename("null"))

        if st.sidebar.checkbox("Summary",value=True):
            st.markdown("""---""")
            st.write("""**data summary**""",df_show.describe().T)

        if st.sidebar.checkbox("Heatmap"):
            st.markdown("""---""")
            fig, ax = plt.subplots()
            sns.heatmap(df_plot.corr(), ax=ax, cmap=cmap,annot=True,vmin=-1, vmax=1,annot_kws={"size":6})
            ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
            st.write("""**heatmap**""",fig)

        if st.sidebar.checkbox("Pairplot"):
            st.markdown("""---""")
            st.write("""**pairplot (wait a minute)**""")
            st.pyplot(sns.pairplot(df_plot,diag_kind='kde',height=2,hue=target_name, corner=True,palette="Accent"))

        if st.sidebar.checkbox("Imbalance check"):
            st.markdown("""---""")
            st.write("""**Imbalance check**""")
            fig2, ax2 = plt.subplots()
            sns.countplot(target_name, data=df_plot,ax=ax2, palette="Set3")
            st.pyplot(fig2)
    except:
        pass
