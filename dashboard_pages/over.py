import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
st.set_option('deprecation.showPyplotGlobalUse', False)
from scripts import data_visualizer
from scripts import df_overview
import df_overview as DfOverview
from scripts import data_selector
import plotly.express as px
from io import StringIO


def load_data():
    data = pd.read_csv('./data/my_clean_data.csv')
    return data


def count_values(data, column_name):
    value_counts = data[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, 'counts']
    return value_counts

def app():
    st.title('User Engagments:')
  
    clean_data= load_data()
    HS_man = count_values(clean_data, 'Handset Manufacturer')
    
    st.header('Top 10 handsets used by the customers') 
    st.write(count_values(clean_data,'Handset Type').head(10))
    
    st.header('Top 3 Handset Manufacturers') 
    st.write(count_values(clean_data, 'Handset Manufacturer').head(3))
    
    st.header('Top 5 handsets of the top 3 handset manufacturers') 
    top3_HS_man = clean_data['Handset Manufacturer'].value_counts().head(3).index
    HS_man = clean_data[clean_data["Handset Manufacturer"].isin(top3_HS_man)]

    result = HS_man.groupby('Handset Manufacturer')['Handset Type'].value_counts().groupby(level=0).head(5)

    for manufacturer, types in result.groupby(level=0):
        st.subheader(manufacturer)
        st.table(types.reset_index(name='Count'))
    
    
    st.markdown('#### Here, from the Above result we can understand that:-')

    st.markdown("1. The most used handset model by customer is Huawei B528S-23A which is manufactured by Huawei")
    st.markdown("2. In regard to manufacturing the highest counts of handsets are made by Apple.But, it looks that Apple's Handset are not prefered by customers.This concers Samsung manufacturers too.")
    st.markdown("3. Since, they have high customers to use, Huawei Handsets manufacturers can be recommended to increase the manufacturing capability to increase there acces to customers.")
    
    st.header('Number of xDR sessions of per user') 
    sessions_per_user = data_selector.find_agg(clean_data, 'MSISDN/Number', 'count', 'Bearer Id', False)
    sessions_per_user.rename(columns={'Bearer Id': 'Number of xDR sessions'}, inplace=True)
    st.write(sessions_per_user.head())
    
    st.header('Average session durations per user') 
    avg_session_durations_per_user = clean_data.groupby('MSISDN/Number').agg({'Dur. (ms).1': 'mean'})
    avg_session_durations_per_user.rename(columns={'Dur. (ms).1': 'Average session duration (ms)'}, inplace=True)
    st.write(avg_session_durations_per_user.sort_values(by=['Average session duration (ms)'], ascending=False).head(10))
    
    st.header('Total download (DL) and upload (UL) data per user')
    total_data = clean_data.groupby('MSISDN/Number')[[ 'Total UL (Bytes)', 'Total DL (Bytes)', 'Total Data Volume (Bytes)']].sum()
    #total_data['MSISDN/Number'] = total_data['MSISDN/Number'].astype(str)

    st.write(total_data.nlargest(10, 'Total Data Volume (Bytes)') )