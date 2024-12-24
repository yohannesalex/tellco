import sys, os
import streamlit as st
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
#st.set_option('deprecation.showPyplotGlobalUse', False)
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from scripts import data_visualizer


from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from scripts import data_visualizer
from io import StringIO

@st.cache_data
def load_data():
    data_load_state = st.text('Loading data...')
    data = pd.read_csv('./data/my_clean_data.csv')
    data_load_state.text('Loading data... done!')     
    return data

@st.cache_data
def load_engagement_data():
    data_load_state = st.text('Loading engagement data...')
    data = pd.read_csv("./data/TellCo_user_engagements.csv")
    data_load_state.text('Loading engagement data... done!')
    return data

@st.cache_data
def user_experiance_data():
    data_load_state = st.text('Loading user experience data...')
    data = pd.read_csv("./data/TellCo_user_experience_data.csv")
    data_load_state.text('Loading user experience data... done!')
    return data
def data_info(df):
    st.markdown('**_______________________________________**')
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
def load_eng_model():
    data_load_state = st.text('Loading model...')
    with open("./models/TellCo_user_engagement.pkl", "rb") as f:
        kmeans1 = pickle.load(f)
        kmeans1.n_init='auto'
        data_load_state.text('Loading the engagement model Done!')         
    return kmeans1

def load_exp_model():
    data_load_state = st.text('Loading model...')
    with open("./models/TellCo_user_experiance.pkl", "rb") as f:
        kmeans2 = pickle.load(f)
        kmeans2.n_init='auto'
        data_load_state.text('Loading the experience model Done!')       
    return kmeans2    
def app():
    st.title('User Satisfaction Analysis')
    st.markdown('##### Load Clean data')
    # load data
    #data_load_state = st.text('Loading data...')
    data = load_data()
    #data_load_state.text('Loading data... done!') 
    data_info(data)

    st.markdown('##### Load engagement data')    
     
    user_engagements = load_engagement_data()
    data_info(user_engagements)

    st.markdown('##### Load experience data')
     
    user_experiance = user_experiance_data()
    data_info(user_experiance)
    
    # load models
    
    kmeans1 = load_eng_model()
    less_engagement = 3
    # Distance between the centroid and samples
    eng_df = user_engagements.set_index('MSISDN/Number')[['time_duration', 'Total Data Volume (Bytes)', 'user_sessions']]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(eng_df)
    st.markdown("**Scaled engagement data**")
    st.write(pd.DataFrame(scaled_array).head(5))    
    
    st.markdown('**Normalized engagement data for the model**')
    data_normalized = normalize(scaled_array)
    st.write(pd.DataFrame(data_normalized).head(5))
    
    #check from the centeroid
    distance = kmeans1.fit_transform(data_normalized)
    distance_from_less_engagement = list(map(lambda x: x[less_engagement], distance))
    user_engagements['engagement_score'] = distance_from_less_engagement
    st.markdown('**Sample engagement data on clusters**')
    st.write(user_engagements.head(5))
    
    st.markdown('# Here, Considering the experience score as the Euclidean distance between the user data point & the worst experience’s cluster members')
    kmeans2 = load_exp_model()
    
    worst_experiance = 0
    exp_df = user_experiance.set_index('MSISDN/Number')[['total_avg_rtt', 'total_avg_tp', 'total_avg_tcp']]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(exp_df)
    st.markdown("**Scaled experiance data**")
    st.write(pd.DataFrame(scaled_array).head(5)) 
    
    data_normalized = normalize(scaled_array)
    st.markdown("**Normalized experiance data**")
    st.write(pd.DataFrame(data_normalized).head(5))
    
    
    exp_df = user_experiance.set_index('MSISDN/Number')
    distance = kmeans2.fit_transform(data_normalized)
    distance_from_worest_experiance = list(map(lambda x: x[worst_experiance], distance))
    user_experiance['experience_score'] = distance_from_worest_experiance
    st.markdown('**Sample Score data**')
    st.write(user_experiance.head(5))
    
    st.markdown('* Consider the average of **both** engagement & experience scores as the satisfaction score & report the top 10 satisfied customer')
    
    user_id_engagement = user_engagements['MSISDN/Number'].values
    user_id_experiance = user_experiance['MSISDN/Number'].values
    user_intersection = list(set(user_id_engagement).intersection(user_id_experiance))
    st.write('**Sample MSISDN/Numbers of engagement & experience scores**' , user_intersection[:5])
    
    user_engagement_df = user_engagements[user_engagements['MSISDN/Number'].isin(user_intersection)]
    
    st.write('* **Engagements on the intersection**', user_engagement_df.shape)
    
    user_experiance_df = user_experiance[user_experiance['MSISDN/Number'].isin(user_intersection)]
    #user_experiance_df.shape
    st.write('* **Experiences on the intersection**', user_experiance_df.shape)
    
    user_df = pd.merge(user_engagement_df, user_experiance_df, on='MSISDN/Number')
    user_df['satisfaction_score'] = (user_df['engagement_score'] + user_df['experience_score'])/2
    st.write('* **Merge Result of engagements and experiences:**',user_df.head(5))
    
    sat_score_df = user_df[['MSISDN/Number', 'engagement_score', 'experience_score', 'satisfaction_score']]
    sat_score_df = sat_score_df.set_index('MSISDN/Number')
    st.write('* **All Scores together sample**', sat_score_df.head(5))
    
    sorted_by_satisfaction = sat_score_df.sort_values('satisfaction_score', ascending=False)
    sat_top_10 = sorted_by_satisfaction['satisfaction_score'].head(10)
    
    image = open("./plotes/Top 10 Satisfied.png", "rb")
    image_bytes = image.read()
    st.markdown('**Top 10 Satisfied Customers**')
# Display the image on the Streamlit dashboard
    st.image(image_bytes, caption='Top 10 Satisfied Customers', use_column_width=True)
    
    st.markdown('**Scatter plot of Engagement with Experience and Satisfaction**')
    fig2 = px.scatter(sat_score_df, 'engagement_score','experience_score', 'satisfaction_score')
    
    fig=go.Figure(fig2)
    #fig.show()
    st.plotly_chart(fig)
    
    
    #fig = data_visualizer.hist(sat_top_10)
    #st.plotly_chart(fig)
        
    #if st.checkbox('Top 10 Satisfied Customers'):
    #    fig = data_visualizer.hist(sat_top_10)
    #    st.plotly_chart(fig)
        
        
    st.write('##### Build a regression model of your choice to predict the satisfaction score of a customer.')    
    st.write('* From Scatter plot we can clearly see when experience score and engagement score increase, satisfaction score will also increase.')
    
    X = sat_score_df[['engagement_score', 'experience_score']]
    y = sat_score_df[['satisfaction_score']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    
    linear_reg = LinearRegression()
    model = linear_reg.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    st.write('**Coefficients:** \n', model.coef_)
    st.write("**Mean squared error:** %.2f" % np.mean((model.predict(X_test) - y_test) ** 2))
    st.write('**Variance score:** %.2f' % model.score(X_test, y_test))
    
    
    #Run a k-means(k=2) on the engagement & the experience score .
    user_satisfaction_df = user_df[['MSISDN/Number', 'engagement_score', 'experience_score']].copy()
    user_satisfaction_df = user_satisfaction_df.set_index('MSISDN/Number')
    st.write(user_satisfaction_df.head(5))
    st.write('**check outliers before fit in to the model**')
    
    fig, ax = plt.subplots()
    ax.boxplot(user_satisfaction_df.values)
    ax.set_xticklabels(user_satisfaction_df.columns)
    ax.set_xlabel('Categories')
    ax.set_ylabel('User Satisfaction')

# Display the plot using Streamlit
    
    
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(user_satisfaction_df)
    st.markdown("**Scaled data**")
    st.write(pd.DataFrame(scaled_array).head(5))
    
    data_normalized = normalize(scaled_array)
    st.markdown("**Normalized data**")
    st.write(pd.DataFrame(data_normalized).head(5))
    
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data_normalized)
    
    st.write(kmeans.labels_)
    
    
    user_satisfaction_df.insert(0, 'cluster', kmeans.labels_)
    st.write('**Satisfaction data**', user_satisfaction_df.head(5))
    st.write('**Satisfaction counts on two clusters**', user_satisfaction_df['cluster'].value_counts())
    
    #fig = px.scatter(user_satisfaction_df, x='engagement_score', y="experience_score", color='cluster')

    #image_bytes = pio.to_image(fig, format='png', width=1200)
    #Image(pio.to_image(fig, format='png', width=1200))
    fig = px.scatter(user_satisfaction_df, x='engagement_score', y='experience_score',
                 color='cluster')#, size='satisfaction_score')
    st.plotly_chart(fig)
    
    #st.image(Image.open(BytesIO(image_bytes)))
    if st.button('Save Satisfaction Data as CSV'):
        user_satisfaction_df.to_csv('../data/TellCo_user_satisfaction.csv')
    
    st.write('**Aggregate the average satisfaction & experience score per cluster.**')
    
    st.write(user_satisfaction_df.groupby('cluster').agg({'engagement_score': 'sum', 'experience_score': 'sum'}))
    
    st.write(' **Note:** - Cluster 1 has higher Engagement and satisfaction score. \n - Cluster 2 has vert low expirience score but higher engagement score.')
    #st.write('Export the final table containing all user id + engagement, experience & satisfaction scores in your local MySQL database.')# Report a screenshot of a select query output on the exported table.
    #engine = create_engine('mysql+pymysql://root:2203@localhost/telecom_user_db')
    
    #Model deployment tracking - deploy the model and monitor your model. Here you can use MlOps tools which can help you to track your model’s change. Your model tracking report includes code version, start and end time, source, parameters, metrics(loss convergence) and artifacts or any output file regarding each specific run. (CSV file, screenshot)
    
    
#if __name__ == '__main__':
#    main()
