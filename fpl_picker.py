from numpy.core.numeric import full
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os
import base64 
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import seaborn as sns


st.set_page_config(layout="wide")

with st.expander('Data Prep'):

    file_location_2022='C:/Users/Darragh/Documents/Python/premier_league/raw_data_2022.csv'
    file_location_2021='C:/Users/Darragh/Documents/Python/premier_league/raw_data_2021.csv'
    file_location_2020='C:/Users/Darragh/Documents/Python/premier_league/raw_data_2020.csv'

    url2022 = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/players_raw.csv'
    url2021 = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/players_raw.csv'
    url2020 = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/players_raw.csv'

    col_selection_url=['id','element_type','first_name','second_name','team']
    @st.cache
    def read_data(x,col_selection):
        return pd.read_csv(x,usecols=col_selection)

    col_selection_week=['team','bps','bonus','player_id','ict_index','minutes','opponent_team','selected','total_points','transfers_in','transfers_out',
    'value','week','year']
    url_csv_2022=read_data(url2022,col_selection_url)
    url_csv_2021=read_data(url2021,col_selection_url)
    df_week_data_raw_2022 = read_data(file_location_2022,col_selection_week)
    df_week_data_raw_2021 = read_data(file_location_2021,col_selection_week)

    @st.cache
    def prep_base_data(url_csv, pick):
        url_csv = url_csv.rename(columns = {'id':'player_id','element_type':'Position'})
        url_csv['Position'] = url_csv['Position'].map({1: 'GK', 2: 'DF', 3:'MD', 4:'FW'})
        url_csv['full_name'] = (url_csv['first_name']+'_'+url_csv['second_name']).str.lower()
        pick_data = pick.rename(columns = {'total_points':'week_points'})
        return pd.merge(url_csv,pick_data, on='player_id',how ='outer')

    @st.cache(suppress_st_warning=True)
    def data_2022_team_names(file):
        file['team'] = file['team'].map({1: 'Arsenal', 2: 'Aston_Villa', 3: 'Brentford', 4:'Brighton', 5:'Burnley',6:'Chelsea',7:'Crystal_Palace',8:'Everton',
        9:'Leicester',10:'Leeds_Utd',11:'Liverpool',12:'Man_City',13:'Man_Utd',14:'Newcastle',15:'Norwich',16:'Southampton',17:'Spurs',
        18:'Watford',19:'West_Ham',20:'Wolves'})
        return file

    @st.cache(suppress_st_warning=True)
    def data_2021_team_names(file):
        file['team'] = file['team'].map({1: 'Arsenal', 2: 'Aston_Villa', 3:'Brighton', 4:'Burnley',5:'Chelsea',6:'Crystal_Palace',7:'Everton',
        8:'Fulham',9:'Leicester',10:'Leeds_Utd',11:'Liverpool',12:'Man_City',13:'Man_Utd',14:'Newcastle',15:'Sheffield_Utd',16:'Southampton',17:'Spurs',
        18:'West_Brow',19:'West_Ham',20:'Wolves'})
        return file

    data_2022 = (data_2022_team_names( (prep_base_data(url_csv_2022, df_week_data_raw_2022)).rename(columns = {'team_x':'team'})))
    data_2021 = (data_2021_team_names( (prep_base_data(url_csv_2021, df_week_data_raw_2021)).rename(columns = {'team_x':'team'})))
    st.write(data_2022.head())

