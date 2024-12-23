import streamlit as st
import sys, os
import sys
sys.path.append('.')
sys.path.append(os.path.abspath(os.path.join('./scripts')))

from pages import Data_Overview
#from pages import user_engagement
from pages import experiance_new
#from pages import user_satisfaction
#from pages import user_experience
from pages import over
from pages import sat
from pages import Business

PAGES = {
    "Home" : Business,
    "Data Overview": Data_Overview,
    "User Engagement Analysis2":  over,
    "User Experience Analytics": experiance_new,
    "User Satisfaction Analysis": sat,
    #"User Engagement Analysis":  user_engagement,
    #"User Experience Analytics2": user_experience,
}

selection = st.sidebar.radio("Go to page", list(PAGES.keys()))
page = PAGES[selection]
page.app()
