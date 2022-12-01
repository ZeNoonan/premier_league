import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import seaborn as sns

st.set_page_config(layout="wide")

def read_file(file,league='la_liga',season=2023):
    df=pd.read_csv(file)
    df['league']=league
    df['season']=season
    return df

factor_la_liga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_factor_result_2022_2023.csv','la_liga',2023)
factor_la_liga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_factor_result_2021_2022.csv','la_liga',2022)
factor_premier_league_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_factor_result_2021_2022.csv','premier_league',2022)
factor_premier_league_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_factor_result_2021_2022.csv','premier_league',2023)
factor_serie_a_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2022_2023.csv','serie_a',2023)
factor_serie_a_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2021_2022.csv','serie_a',2022)
factor_bundesliga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2022_2023.csv','bundesliga',2023)
factor_bundesliga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2021_2022.csv','bundesliga',2022)

st.write(factor_bundesliga_2021_2022)
# combine file above into one dataframe
combined_df=pd.concat([factor_la_liga_2022_2023,factor_la_liga_2021_2022,factor_premier_league_2021_2022])