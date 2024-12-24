import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
st.set_option('deprecation.showPyplotGlobalUse', False)
from scripts import data_visualizer
#from scripts import df_overview
#import df_overview as DfOverview
from scripts.outlier_handler import OutlierHandler
import plotly.express as px
from io import StringIO
from sklearn.preprocessing import StandardScaler, normalize
import pickle
from PIL import Image
from sklearn.cluster import KMeans
@st.cache_data
def load_data():
    data = pd.read_csv('./data/my_clean_data.csv')
    return data


def count_values(data, column_name):
    value_counts = data[column_name].value_counts().reset_index()
    value_counts.columns = [column_name, 'counts']
    return value_counts
@st.cache_data
def getExperienceDataFrame():
    df = load_data().copy()
    user_experience_df = df[[
        "MSISDN/Number",
        "Avg RTT DL (ms)",
        "Avg RTT UL (ms)",
        "Avg Bearer TP DL (kbps)",
        "Avg Bearer TP UL (kbps)",
        "TCP DL Retrans. Vol (Bytes)",
        "TCP UL Retrans. Vol (Bytes)",
        "Handset Type"]].copy()
    
    user_experience_df['total_avg_rtt'] = user_experience_df['Avg RTT DL (ms)'] + user_experience_df['Avg RTT UL (ms)']
    user_experience_df['total_avg_tp'] = user_experience_df['Avg Bearer TP DL (kbps)'] + user_experience_df['Avg Bearer TP UL (kbps)']
    user_experience_df['total_avg_tcp'] = user_experience_df['TCP DL Retrans. Vol (Bytes)'] + user_experience_df['TCP UL Retrans. Vol (Bytes)']

    return user_experience_df
@st.cache_data
def getExperienceData():
    df = getExperienceDataFrame().copy()
    user_experience = df.groupby('MSISDN/Number').agg({
        'total_avg_rtt': 'sum',
        'total_avg_tp': 'sum',
        'total_avg_tcp': 'sum'})
    return user_experience


        
def app():
    st.title('User Experience Analytics:')
  
    #st.title('User Experience Analytics')
    #st.header("Top 10 customers per engagement metrics")
    user_experience = getExperienceDataFrame().copy()
    st.markdown('#### User Experience Metrics Data Info:')
    buffer = StringIO()
    user_experience.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    #st.write(user_experience.info())
    st.markdown(' **1. Aggregate, per customer, the following information (treat missing & outliers by replacing by the mean or the mode of the corresponding variable):**')
    st.markdown('* Average TCP retransmission')
    st.markdown('* Average RTT')
    st.markdown('* Handset type')
    st.markdown('* Average throughput')
    
    
    _user_experience = user_experience.groupby('MSISDN/Number').agg({
    'total_avg_rtt': 'sum',
    'total_avg_tp': 'sum',
    'total_avg_tcp': 'sum',
    'Handset Type': [lambda x: x.mode()[0]]})

    user_experience = pd.DataFrame(columns=["total_avg_rtt", "total_avg_tp", "total_avg_tcp", "Handset Type"])

    user_experience["total_avg_rtt"] = _user_experience["total_avg_rtt"]['sum']
    user_experience["total_avg_tp"] = _user_experience["total_avg_tp"]['sum']
    user_experience["total_avg_tcp"] = _user_experience["total_avg_tcp"]['sum']
    user_experience["Handset Type"] = _user_experience["Handset Type"]['<lambda>']
    st.write('**Aggregating User Experience Metrics Per User**')
    
    user_experience['MSISDN/Number'] = user_experience['MSISDN/Number'].astype(str)
    st.write(user_experience.head())  
    
    st.markdown(' 2. Compute & list 10 of the top, bottom and most frequent:')
    st.markdown('* TCP values in the dataset.')
    st.markdown('* RTT values in the dataset.')
    st.markdown('* Throughput values in the dataset')
    
    #TCP values in the dataset.
        
    st.write("**TCP values in the dataset**")
    image = Image.open('./data/first.jpg')
    st.image(image, caption='TCP values in the dataset', use_column_width=True)
              
    st.write("**RTT values in the dataset.**")
    
    image = Image.open('./data/second.jpg')
    st.image(image, caption='RTT values in the dataset', use_column_width=True)
    
    st.write("**Throughput values in the dataset**")
    
    image = Image.open('./data/third.jpg')
    st.image(image, caption='Throughput values in the dataset', use_column_width=True)
    
    st.markdown(' 3. Compute & report:')
    
    st.markdown('* The distribution of the average throughput per handset type and provide interpretation for your findings.')
    st.markdown('* The average TCP retransmission view per handset type and provide interpretation for your findings.')
    
         # Assuming handset_type_df is your DataFrame
    handset_type_df = user_experience.groupby('Handset Type').agg({'total_avg_tp': 'mean', 'total_avg_tcp': 'mean'})

    # Display the first few rows of the aggregated DataFrame
    st.write(handset_type_df.head())
    
    image = Image.open('./data/fourth.jpg')
    st.image(image, caption='total_avg_tp', use_column_width=True)   
    
    
    image = Image.open('./data/fivth.jpg')
    st.image(image, caption='total_avg_tp top 20', use_column_width=True) 
   

    image = Image.open('./data/6.jpg')
    st.image(image, caption='Total Average Throughput', use_column_width=True) 
    
    
    image = Image.open('./data/7.jpg')
    st.image(image, caption='Total Average Throughput', use_column_width=True) 

   
    
    
    
    
    #The average TCP retransmission view per handset type and provide interpretation for your findings.
    sorted_by_tcp = handset_type_df.sort_values('total_avg_tcp', ascending=False)
    top_tcp = sorted_by_tcp['total_avg_tcp']
   
    
    st.markdown(' 4. Using the experience metrics above, perform a k-means clustering (where k = 3) to segment users into groups of experiences and provide a brief description of each cluster. (The description must define each group based on your understanding of the data)')
    experience_metric_df = user_experience[["total_avg_rtt", "total_avg_tp", "total_avg_tcp"]].copy()
    st.write(experience_metric_df.head())
    
    df_outliers = OutlierHandler(experience_metric_df)
    df_outliers.replace_outliers_with_fences(
    ["total_avg_rtt",
     "total_avg_tp",
     "total_avg_tcp"])

    df_outliers.getOverview(["total_avg_rtt","total_avg_tp", "total_avg_tcp"])
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_outliers.df)
    
    #pd.DataFrame(scaled_array).head(5)
    data_normalized = normalize(scaled_array)
    #pd.DataFrame(data_normalized).head(5)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_normalized)
    kmeans.labels_
    
    experience_metric_df.insert(0, 'cluster', kmeans.labels_)
    
    st.write(experience_metric_df['cluster'].value_counts())
    
    fig = px.scatter(experience_metric_df, x='total_avg_rtt', y="total_avg_tp",
                 color='cluster', size='total_avg_tcp')
    fig.show()

    
    fig = go.Figure(data=[go.Scatter3d(x=experience_metric_df['total_avg_tcp'], 
                                       y=experience_metric_df['total_avg_rtt'], 
                                       z=experience_metric_df['total_avg_tp'], 
                                       mode='markers',marker=dict(color=experience_metric_df['cluster']))])

    fig.update_layout(scene=dict(xaxis_title='Total Average TCP',
                             yaxis_title='Total Average RTT',
                             zaxis_title='Total Average TP'),
                  width=900,  # Set the width of the plot
                  height=800,  # Set the height of the plot
                  title='3D Scatter Plot')

    fig.show()

# Save the final data frame

    if st.button('Save CSV'):
        helper.save_csv(user_experience,'./data/TellCo_user_experience_data.csv', index=True)

        with open("./models/TellCo_user_experiance.pkl", "wb") as f:
            pickle.dump(kmeans, f)


