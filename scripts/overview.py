import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
sys.path.append(os.path.abspath(os.path.join('../scripts..')))
from data_visualizer import *
from data_selector import *
#sys.path.append('../scripts/')
from scripts import outlier_handler

import constants as con
from outlier_handler import OutlierHandler
from df_overview import DfOverview
# from scripts import df_overview

def loadDescription():
    df = pd.read_excel("./data/Field_Descriptions.xlsx")
    return df


def loadOriginalData():
    df = pd.read_csv("./data/Week1_challenge_data_source(CSV).csv")
    return df


def loadPreprocessedData():
    df = pd.read_csv("./data/clean_data.csv")
    #print(df.shape)
    return df


def app():
    st.title('Data Overview')

    st.header('Table Description')
    st.markdown(
    '''
        The telecom dataset has 150001 observations with 55 features. 
        Here is description of all the features
    ''')
    df = loadDescription()
    st.write(df)

    st.header('Here is sample data from the table')
    df = loadOriginalData()
    st.write(df.head(10))

    st.header('Detailed Information On the Dataset')
    st.markdown(
        '''
    The table below shows that:
    - Count of unique values in each columns
    - Persentage of unique values in each columns
    - Count of None values in each columns
    - Persentage of None values in each columns
    - Min, Max, and Median values in each columns
    ''')
    overview = DfOverview(df)
    dfOverview = overview.getOverview()
    st.write(dfOverview)

    st.header('Outliers in the data')
    df = loadPreprocessedData()
    numeric_df = df[con.NUMERIC_COLUMNS].copy()
    st.markdown(
    '''
    The table below shows outliers in the data.

    The table contains:
    - IQR for each columns
    - skew for each columns
    - Count of Outliers in each columns
    - Persentage of Outliers in each columns
    - Min, Max, Q1, median and Q3 for each columns
    ''')
    
    
    df_outliers = DfOutlier(numeric_df)
    overview = df_outliers.getOverview()
    overview.sort_values(by=["number_of_outliers"], inplace=True)
    st.write(overview)
