import sys
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
#from streamlit_plot import *
from sklearn.cluster import KMeans

sys.path.insert(1, '../scripts')
from constants import *
from df_outlier import DfOutlier
from df_overview import DfOverview
from df_utils import DfUtils
from df_helper import DfHelper
from streamlit_plot import *

utils = DfUtils()
helper = DfHelper()


st.cache_data
def getEngagemetData():
    df = pd.read_csv("./data/TellCo_user_engagements.csv")
    print(df.info())
    return df


st.cache_data
def getExperienceData():
    df = pd.read_csv("./data/TellCo_user_experience_data.csv")
    print(df.info())
    return df


st.cache_data
def getNormalExperience(df):
    df_outliers = DfOutlier(df.copy())
    cols = ["total_avg_rtt",
            "total_avg_tp",
            "total_avg_tcp"]
    df_outliers.replace_outliers_with_iqr(cols)
    df = utils.scale_and_normalize(df_outliers.df)
    return df

st.cache_data
def getNormalEngagement(df):
    df_outliers = DfOutlier(df.copy())
    cols = ['user_sessions', 'time_duration', 'Total Data Volume (Bytes)']
    df_outliers.replace_outliers_with_iqr(cols)
    res_df = utils.scale_and_normalize(df_outliers.df)
    return res_df


def getEngagemetModel():
    with open("./models/TellCo_user_engagement.pkl", "rb") as f:
        kmeans = pickle.load(f)
        kmeans.n_init='auto'
    return kmeans

def getExperienceModel():
    with open("./models/TellCo_user_experiance.pkl", "rb") as f:
        kmeans = pickle.load(f)
        kmeans.n_init='auto'
    return kmeans

def getUserEngagement(less_engagement):
    eng_df = getEngagemetData().copy()
    eng_model = getEngagemetModel()
    eng_df = eng_df.set_index('MSISDN/Number')[['time_duration', 'Total Data Volume (Bytes)', 'user_sessions']]
    eng_normal = getNormalEngagement(eng_df).copy()
    
    
    distance = eng_model.fit_transform(eng_normal)
    distance_from_less_engagement = list(
        map(lambda x: x[less_engagement], distance))
    eng_df['engagement_score'] = distance_from_less_engagement
    return eng_df


def getUserExperience(worst_experience):
    exp_df = getExperienceData().copy()
    exp_model = getExperienceModel()
    print(exp_df.info())
    exp_df = exp_df.set_index('MSISDN/Number')[ ['total_avg_rtt', 'total_avg_tp', 'total_avg_tcp']]

    exp_normal = getNormalExperience(exp_df).copy()
    distance = exp_model.fit_transform(exp_normal)
    distance_from_worst_experience = list(
        map(lambda x: x[worst_experience], distance))
    exp_df['experience_score'] = distance_from_worst_experience
    return exp_df

def getSatisfactionData(less_engagement, worst_experience):
    user_engagement = getUserEngagement(less_engagement)
    user_experience = getUserExperience(worst_experience)

    user_engagement.reset_index(inplace=True)
    user_experience.reset_index(inplace=True)

    user_id_engagement = user_engagement['MSISDN/Number'].values
    user_id_experience = user_experience['MSISDN/Number'].values

    user_intersection = list(
        set(user_id_engagement).intersection(user_id_experience))

    user_engagement_df = user_engagement[user_engagement['MSISDN/Number'].isin(
        user_intersection)]
    
    user_experience_df = user_experience[user_experience['MSISDN/Number'].isin(
        user_intersection)]

    user_df = pd.merge(user_engagement_df, user_experience_df, on='MSISDN/Number')
    user_df['satisfaction_score'] = (
        user_df['engagement_score'] + user_df['experience_score'])/2
    sat_score_df = user_df[['MSISDN/Number', 'engagement_score',
                            'experience_score', 'satisfaction_score']]
    sat_score_df = sat_score_df.set_index('MSISDN/Number')
    return sat_score_df


def app():
    st.title('User Satisfaction Analysis')
    num1 = st.sidebar.selectbox('Select the cluster with less Engagement', range(0, 20))
    num2 = st.sidebar.selectbox(
        'Select the cluster with worst Experience', range(0, 20))
    if st.sidebar.button('Ok'):
        df = getSatisfactionData(num1, num2)

        st.header("Top 10 customers per engagement metrics")
        sorted_by_satisfaction = df.sort_values(
            'satisfaction_score', ascending=False)
        sat_top_10 = sorted_by_satisfaction['satisfaction_score'].head(10)
        hist(sat_top_10)

        st.markdown(
        '''
            Plot showing relationship between engagement score and satisfaction score.        
        ''')
        scatter(df, 'engagement_score',
                'experience_score', 'satisfaction_score')
