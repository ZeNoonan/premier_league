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

    col_selection_url=['id','element_type','first_name','second_name']
    col_selection_url_2020=['id','element_type','first_name','second_name','team']
    # st.write('url master data 2020', pd.read_csv(url2020).head())

    @st.cache(suppress_st_warning=True)
    def read_data(x,col_selection):
        return pd.read_csv(x,usecols=col_selection)

    @st.cache(suppress_st_warning=True)
    def data_2020_team_names(file):
        file['team'] = file['team'].map({1: 'Arsenal', 2: 'Aston_Villa', 3:'Bournemouth', 4:'Brighton',5:'Burnley',6:'Chelsea',7:'Crystal_Palace',
        8:'Everton',9:'Leicester',10:'Liverpool',11:'Man_City',12:'Man_Utd',13:'Newcastle',14:'Norwich',15:'Sheffield_Utd',16:'Southampton',17:'Spurs',
        18:'Watford',19:'West_Ham',20:'Wolves'})
        return file

    col_selection_week=['team','bps','bonus','player_id','ict_index','minutes','opponent_team','selected','total_points','transfers_in','transfers_out',
    'value','week','year']
    col_selection_week_2020=['bps','bonus','player_id','ict_index','minutes','opponent_team','selected','total_points','transfers_in','transfers_out',
    'value','week','year','round','fixture']
    url_csv_2022=read_data(url2022,col_selection_url)
    url_csv_2021=read_data(url2021,col_selection_url)
    url_csv_2020=read_data(url2020,col_selection_url_2020).copy()
    url_csv_2020=data_2020_team_names(url_csv_2020)
    # url_csv_2020=data_2020_team_names((read_data(url2020,col_selection_url_2020).copy()))
    # st.write('master data 2020',url_csv_2020.head())
    df_week_data_raw_2022 = read_data(file_location_2022,col_selection_week)
    df_week_data_raw_2021 = read_data(file_location_2021,col_selection_week)
    # st.write('2020 raw data weekly',pd.read_csv(file_location_2020).head())
    df_week_data_raw_2020 = read_data(file_location_2020,col_selection_week_2020)
    # st.write(df_week_data_raw_2022.head())

    @st.cache
    def prep_base_data(url_csv, pick):
        url_csv = url_csv.rename(columns = {'id':'player_id','element_type':'Position'})
        url_csv['Position'] = url_csv['Position'].map({1: 'GK', 2: 'DF', 3:'MD', 4:'FW'})
        url_csv['full_name'] = (url_csv['first_name']+'_'+url_csv['second_name']).str.lower()
        pick_data = pick.rename(columns = {'total_points':'week_points'})
        return pd.merge(url_csv,pick_data, on='player_id',how ='outer')

    @st.cache(suppress_st_warning=True)
    def clean_blank_gw(x,team1,team2,week_no):
        x = x [ ((x ['team']==team1) | (x ['team']==team2)) & (x['week']==week_no) ].copy()
        x['round']=week_no + 1
        x['week'] =week_no + 1
        x['minutes']=np.NaN
        x['week_points']=np.NaN
        x['fixture'] =np.NaN
        return x

    @st.cache(suppress_st_warning=True)
    def data_2020_clean_double_gw(url_pick2020):
        url_pick2020=url_pick2020[ ~(url_pick2020['round']==29) | ~(url_pick2020['fixture']==275)]
        url_pick2020.loc[:,'week']=url_pick2020['week'].replace({39:30,40:31,41:32,42:33,43:34,44:35,45:36,46:37,47:38})
        url_pick2020.loc[:,'round']=url_pick2020['round'].replace({39:30,40:31,41:32,42:33,43:34,44:35,45:36,46:37,47:38})
        gw_18_blank = clean_blank_gw(url_pick2020,10,19,17)
        gw_28_blank = clean_blank_gw(url_pick2020,11,3,27)
        gw_28_blank_1 = clean_blank_gw(url_pick2020,2,15,27)
        return pd.concat([url_pick2020,gw_18_blank,gw_28_blank,gw_28_blank_1])    

    data_2022 = (( (prep_base_data(url_csv_2022, df_week_data_raw_2022))))
    data_2021 = (( (prep_base_data(url_csv_2021, df_week_data_raw_2021))))
    data_2020 = (( (prep_base_data(url_csv_2020, df_week_data_raw_2020)))).copy()
    data_2020 = data_2020_clean_double_gw(data_2020)

    @st.cache(suppress_st_warning=True)
    def combine_dataframes(a,b,c):
        return pd.concat ([a,b,c], axis=0,sort = True)

    full_df = combine_dataframes(data_2022,data_2021,data_2020).drop(['fixture'],axis=1).copy()
    

    # st.write(data_2022.head())
    # st.write(data_2021.head())
    # st.write(data_2020[data_2020['full_name'].str.contains('salah')])

    @st.cache(suppress_st_warning=True)
    def column_calcs_1(df):
        df['Price'] =df['value'] / 10
        df['Game_1'] = np.where((df['minutes'] > 0.5), 1, 0)
        df['Clean_Pts'] = np.where(df['Game_1']==1,df['week_points'], np.NaN) # setting a slice on a slice - just suppresses warning....
        return df.sort_values(by=['full_name', 'year', 'week'], ascending=[True, True, True]) # THIS IS IMPORTANT!! EWM doesn't work right unless sorted

    full_df=column_calcs_1(full_df)
    df=full_df.reset_index().rename(columns={'index':'id_merge'})

    @st.cache(suppress_st_warning=True)
    def column_calcs_2(df):
        df_calc=df[df['Game_1']>0].copy()
        df_calc['last_60_games']=df_calc.groupby(['full_name'])['Game_1'].rolling(window=60,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_60_points']=df_calc.groupby(['full_name'])['Clean_Pts'].rolling(window=60,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_60_ppg']=df_calc['last_60_points']/df_calc['last_60_games']
        df=pd.merge(df,df_calc,how='outer')
        df['last_60_ppg']=df['last_60_ppg'].fillna(method='ffill')
        df['last_60_games']=df['last_60_games'].fillna(method='ffill')
        df['games_total'] = df.groupby (['full_name'])['Game_1'].transform('sum')
        return df

    full_df=column_calcs_2(full_df)

    cols_to_move=['full_name','team','week','year','minutes','Clean_Pts','last_60_ppg','games_total','last_60_games','last_60_points',
    'bps','bonus','player_id','ict_index',
    'opponent_team','selected','transfers_in','transfers_out',
    'value']
    cols = cols_to_move + [col for col in full_df if col not in cols_to_move]
    full_df=full_df[cols]
    st.write(full_df[full_df['full_name'].str.contains('bruno miguel')])


   