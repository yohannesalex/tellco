import sys
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_plot import *
from sklearn.cluster import KMeans

sys.path.insert(1, '../scripts')
from df_helper import DfHelper
from df_utils import DfUtils
from df_overview import DfOverview
from df_outlier import DfOutlier
from constants import *
from streamlit_plot import *
from data_visualizer import *
import pyarrow as pa
utils = DfUtils()
helper = DfHelper()


import matplotlib.pyplot as plt
#@st.cache_data
def loadCleanData():
    df = pd.read_csv("./data/my_clean_data.csv")
    return df


#@st.cache_data
def getEngagemetData():
    df = loadCleanData().copy()
    user_engagement_df = df[['MSISDN/Number', 'Bearer Id', 'Dur. (ms).1', 'Total Data Volume (Bytes)']].copy(
    ).rename(columns={'Dur. (ms).1': 'time_duration', 'Total Data Volume (Bytes)': 'Total Data Volume (Bytes)'})
    user_engagement = user_engagement_df.groupby(
        'MSISDN/Number').agg({'Bearer Id': 'count', 'time_duration': 'sum', 'Total Data Volume (Bytes)': 'sum'})
    user_engagement = user_engagement.rename(
        columns={'Bearer Id': 'user_sessions'})
    return user_engagement


#@st.cache_data
def getNormalData(df):
    res_df = utils.scale_and_normalize(df)
    return res_df


#@st.cache_data
def get_distortion_and_inertia(df, num):
    distortions, inertias = utils.choose_kmeans(df.copy(), num)
    return distortions, inertias



def plotTop10(df):
    col = st.sidebar.selectbox(
        "Select top 10 from", (["Sessions", "Duration", "Total data volume"]))
    if col == "Sessions":
        sessions = df.nlargest(10, "user_sessions")['user_sessions']
        return plot_hist(sessions)
    elif col == "Duration":
        duration = df.nlargest(10, "time_duration")['time_duration']
        return hist(duration)
    else:
        total_data_volume = df.nlargest(
            10, "Total Data Volume (Bytes)")['Total Data Volume (Bytes)']
        return hist(total_data_volume)


def elbowPlot(df, num):
    distortions, inertias = get_distortion_and_inertia(df, num)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Distortion", "Inertia")
    )
    fig.add_trace(go.Scatter(x=np.array(range(1, num)),
                             y=distortions), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.array(
        range(1, num)), y=inertias), row=1, col=2)
    fig.update_layout(title_text="The Elbow Method")
    st.plotly_chart(fig)


def app():
    st.title('User Engagement analysis')
    st.header("Top 10 customers per engagement metrics")
    user_engagement = getEngagemetData().copy()
    
    df_outliers = DfOutlier(user_engagement)
    cols = ['user_sessions', 'time_duration', 'Total Data Volume (Bytes)']
    df_outliers.replace_outliers_with_iqr(cols)
    user_engagement = df_outliers.df
    
    
    plotTop10((user_engagement))

    st.header("Clustering customers based on their engagement metric")
    st.markdown(
    '''
        Here we will try to cluster customers based on their engagement.
        To find the optimized value of k, first, let's plot an elbow curve graph.
        To start, choose the number of times to runs k-means.
    ''')
    
    num = st.selectbox('Select', range(0, 20))
    tracemalloc.start()
    select_num = 1
    
    if(num != 0):
        normal_df = getNormalData(user_engagement)
        elbowPlot(normal_df, num+1)

        st.markdown(
        '''
            Select the optimized values for k
        ''')
        select_num = st.selectbox('Select', range(1, num+1))
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        st.write(f"Current memory usage: {current_memory} bytes")
        st.write(f"Peak memory usage: {peak_memory} bytes")
    if(select_num != 1):

        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(normal_df)
        user_engagement.insert(0, 'cluster', kmeans.labels_)

        st.markdown(
        '''
            Number of elements in each cluster
        ''')
        st.write(user_engagement['cluster'].value_counts())

        show2D = False
        if st.button('Show 2D visualization'):
            if(show2D):
                show2D = False
            else:
                show2D = True


        if(show2D):
            st.markdown(
            '''
                2D visualization of cluster
            ''')
            scatter(user_engagement, x='Total Data Volume (Bytes)', y="time_duration",
                    c='cluster', s='user_sessions')
        
        show3D = False
        if st.button('Show 3D visualization'):
            if(show3D == True):
                show3D = False
            else:
                show3D = True
        if(show3D):
            st.markdown(
            '''
                3D visualization of cluster
            ''')
            scatter3D(user_engagement, x="Total Data Volume (Bytes)", y="time_duration", z="user_sessions",
                    c="cluster", interactive=True)
        
        st.warning(
            'Remember cluster with the least engagement. we need that for satisfaction analysis')
        st.markdown(
        '''
            Save the model for satisfaction analysis
        ''')
        if st.button('Save Model'):
            helper.save_csv(user_engagement,
                            './data/user_engagement.csv', index=True)

            with open("./models/user_engagement.pkl", "wb") as f:
                pickle.dump(kmeans, f)
