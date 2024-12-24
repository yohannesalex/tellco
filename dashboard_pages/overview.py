import sys
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

sys.path.insert(1, '../scripts')
from constants import *
from df_outlier import DfOutlier
from df_overview import DfOverview
# from scripts import df_overview
NUMERIC_COLUMNS = ['Start ms',
                   'End ms',
                   'Dur. (ms).1',
                   'Avg RTT DL (ms)',
                   'Avg RTT UL (ms)',
                   'Avg Bearer TP DL (kbps)',
                   'Avg Bearer TP UL (kbps)',
                   'TCP DL Retrans. Vol (Bytes)',
                   'TCP UL Retrans. Vol (Bytes)',
                   'DL TP < 50 Kbps (%)',
                   '50 Kbps < DL TP < 250 Kbps (%)',
                   '250 Kbps < DL TP < 1 Mbps (%)',
                   'DL TP > 1 Mbps (%)',
                   'UL TP < 10 Kbps (%)',
                   '10 Kbps < UL TP < 50 Kbps (%)',
                   '50 Kbps < UL TP < 300 Kbps (%)',
                   'UL TP > 300 Kbps (%)',
                   'Activity Duration DL (ms)',
                   'Activity Duration UL (ms)',
                   'Nb of sec with Vol DL < 6250B',
                   'Nb of sec with Vol UL < 1250B',
                   'Social Media DL (Bytes)',
                   'Social Media UL (Bytes)',
                   'Google DL (Bytes)',
                   'Google UL (Bytes)',
                   'Email DL (Bytes)',
                   'Email UL (Bytes)',
                   'Youtube DL (Bytes)',
                   'Youtube UL (Bytes)',
                   'Netflix DL (Bytes)',
                   'Netflix UL (Bytes)',
                   'Gaming DL (Bytes)',
                   'Gaming UL (Bytes)',
                   'Other UL (Bytes)',
                   'Other DL (Bytes)',
                   'Total UL (Bytes)',
                   'Total DL (Bytes)']

def loadDescription():
    df = pd.read_excel("./data/Field_Descriptions.xlsx")
    return df


def loadOriginalData():
    df = pd.read_csv("./data/Week1_challenge_data_source(CSV).csv")
    return df


def loadPreprocessedData():
    df = pd.read_csv("./data/my_clean_data.csv")
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
    st.write(df, width=1200)

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
    numeric_df = df[NUMERIC_COLUMNS].copy()
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
