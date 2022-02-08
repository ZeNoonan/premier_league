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

future_gameweek=24
current_week=23

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

    full_df = combine_dataframes(data_2022,data_2021,data_2020).drop(['fixture','round'],axis=1).copy()
    

    # st.write(data_2022.head())
    # st.write(data_2021.head())
    # st.write(data_2020[data_2020['full_name'].str.contains('salah')])

    @st.cache(suppress_st_warning=True)
    def column_calcs_1(df):
        df['Price'] =df['value'] / 10
        df['Game_1'] = np.where((df['minutes'] > 0.5), 1, 0)
        df['games_2022'] = np.where((df['year'] == 2022), df['Game_1'], 0)
        df['Clean_Pts'] = np.where(df['Game_1']==1,df['week_points'], np.NaN) # setting a slice on a slice - just suppresses warning....
        return df.sort_values(by=['full_name', 'year', 'week'], ascending=[True, True, True]) # THIS IS IMPORTANT!! EWM doesn't work right unless sorted

    full_df=column_calcs_1(full_df)
    df=full_df.reset_index().rename(columns={'index':'id_merge'})
    # st.write('full df z', df)

    @st.cache(suppress_st_warning=True)
    def column_calcs_2(df):
        df_calc=df[df['Game_1']>0].copy()

        df_calc['last_76_games']=df_calc.groupby(['full_name'])['Game_1'].rolling(window=76,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_76_points']=df_calc.groupby(['full_name'])['Clean_Pts'].rolling(window=76,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_76_ppg']=df_calc['last_76_points']/df_calc['last_76_games']
        df_calc['last_38_games']=df_calc.groupby(['full_name'])['Game_1'].rolling(window=38,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_38_points']=df_calc.groupby(['full_name'])['Clean_Pts'].rolling(window=38,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_38_ppg']=df_calc['last_38_points']/df_calc['last_38_games']
        df_calc['last_19_games']=df_calc.groupby(['full_name'])['Game_1'].rolling(window=19,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_19_points']=df_calc.groupby(['full_name'])['Clean_Pts'].rolling(window=19,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df_calc['last_19_ppg']=df_calc['last_19_points']/df_calc['last_19_games']

        df_calc['games_2022_rolling']=df_calc.groupby(['full_name'])['games_2022'].cumsum()

        df=pd.merge(df,df_calc,how='outer')
        df['last_76_ppg']=df['last_76_ppg'].fillna(method='ffill')
        df['last_76_games']=df['last_76_games'].fillna(method='ffill')
        df['last_38_ppg']=df['last_38_ppg'].fillna(method='ffill')
        df['last_38_games']=df['last_38_games'].fillna(method='ffill')
        df['last_19_ppg']=df['last_19_ppg'].fillna(method='ffill')
        df['last_19_games']=df['last_19_games'].fillna(method='ffill')

        df['games_2022_rolling']=df['games_2022_rolling'].fillna(method='ffill')

        df['games_total'] = df.groupby (['full_name'])['Game_1'].transform('sum')
        return df

    full_df=column_calcs_2(full_df).drop(['value','week_points','first_name','second_name'],axis=1)

    # st.write('check this full df',full_df[full_df['full_name'].str.contains('kane')])

    # cols_to_move=['full_name','Position','Price','team','week','year','minutes','Clean_Pts','last_76_ppg','last_38_ppg','games_total',
    # 'last_76_games','last_76_points',
    # 'games_total','last_38_games','last_38_points','bps','bonus','player_id','ict_index',
    # 'opponent_team','selected','transfers_in','transfers_out']

    cols_to_move=['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','last_76_ppg','last_38_ppg','last_19_ppg','games_total','last_38_games',
    'selected']

    cols = cols_to_move + [col for col in full_df if col not in cols_to_move]
    # st.write('check', full_df[cols])
    full_df=full_df.sort_values(by=['full_name', 'year', 'week'], ascending=[True, False, False])
    full_df=(full_df[cols]).sort_values(by=['full_name', 'year', 'week','games_2022_rolling'], ascending=[True, False, False,False])
    # st.write(full_df[full_df['full_name'].str.contains('bruno miguel')])
    format_mapping={'week':"{:,.0f}",'year':"{:.0f}",'minutes':"{:,.0f}",'Clean_Pts':"{:,.0f}",'last_76_ppg':"{:,.1f}",'games_total':"{:,.0f}",
    'last_76_games':"{:,.0f}",'last_76_points':"{:,.0f}",'Price':"{:,.1f}",'selected':"{:,.0f}",'last_38_ppg':"{:,.1f}",'last_38_games':"{:,.0f}",
    'last_19_games':"{:,.0f}",'last_19_ppg':"{:,.1f}",'games_2022_rolling':"{:,.0f}",'ppg_76_rank':"{:,.0f}",'total_sum_rank':"{:,.0f}",
    'value_ppg':"{:,.0f}",'value_rank':"{:,.0f}",'selected_rank':"{:,.0f}",'transfers_balance':"{:,.0f}",
    'net_transfers_rank':"{:,.0f}",'totals_ranked':"{:,.0f}"}

with st.expander('Player Detail by Week'):
    player_names_pick=full_df['full_name'].unique()
    names_selected_pick = st.selectbox('Select players',player_names_pick, key='player_pick',index=0)
    player_selected_detail_by_week = full_df[full_df['full_name']==names_selected_pick]
    st.write(player_selected_detail_by_week.style.format(format_mapping))



with st.expander('Player Stats Latest'):

    # @st.cache(suppress_st_warning=True)
    def find_latest_player_stats(x):
        week_no = st.number_input ("Week number?", min_value=int(1),value=int(21))
        x=x[x['week'] == week_no]
        x = x.sort_values(by=['year', 'week'], ascending=[False, False]).drop_duplicates('full_name')
        # x = x[x['games_2022_rolling']>2] # want to exclude players who haven't played at all or less than once in 2022 season
        return x[x['year'] == 2022]

    latest_df = find_latest_player_stats(full_df)
    
    latest_df=latest_df.sort_values(by=['last_76_ppg'],ascending=False)
    
    def ranked_players(x):
        # only want players who played greater than a season ie 38 games big sample size
        x = x[x['games_total']>38]
        x['ppg_76_rank']=x.loc[:,['last_76_ppg']].rank(method='dense', ascending=False)
        return x

    def value_rank(x):
        x['total_selected']=9000000
        x['%_selected']=x['selected'] / x['total_selected']
        x['value_ppg']=x['last_76_ppg']/(x['%_selected']+1)
        # x['value_ppg']=x['last_76_ppg']/(x['%_selected'])
        x['value_rank']=x.loc[:,['value_ppg']].rank(method='dense', ascending=False)
        x['selected_rank']=x.loc[:,['%_selected']].rank(method='dense', ascending=False)
        return x

    latest_df = ranked_players(latest_df)
    latest_df = value_rank(latest_df)

    weekly_transfers_in=read_data('C:/Users/Darragh/Documents/Python/premier_league/week_transfers_in.csv',col_selection=['full_name','transfers_balance'])
    def merge_latest_transfers(x):
        return pd.merge(x,weekly_transfers_in,on=['full_name'],how='left')

    def weekly_transfers_historical(x):
        x['transfers_balance']=x['transfers_in']-x['transfers_out']
        return x

    def rank_calc(x):
        # USE THIS FOR LATEST TRANSFERS IN WEEK
        x['net_transfers_rank']=x.loc[:,['transfers_balance']].rank(method='dense', ascending=False)
        return x

    def rank_total_calc(x):
        col_list_1=['ppg_76_rank','value_rank','net_transfers_rank']
        x['total_sum_rank']=x[col_list_1].sum(axis=1)
        x['totals_ranked']=x.loc[:,['total_sum_rank']].rank(method='dense', ascending=True)
        return x

    # latest_df = merge_latest_transfers(latest_df)

    latest_df = weekly_transfers_historical(latest_df)
    latest_df = rank_calc(latest_df)
    latest_df = rank_total_calc(latest_df)

    cols_to_move=['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','totals_ranked','total_sum_rank',
    'ppg_76_rank','value_rank','net_transfers_rank','last_76_ppg','value_ppg','selected_rank','transfers_balance',
    'last_38_ppg','last_19_ppg','games_total','last_38_games','selected']
    cols = cols_to_move + [col for col in latest_df if col not in cols_to_move]
    latest_df=((latest_df[cols].sort_values(by=['totals_ranked'],ascending=True)))
    st.write('number of players', latest_df['full_name'].count())
    st.write(latest_df.set_index('full_name').style.format(format_mapping))

    goalkeeper_data=latest_df[(latest_df['Position']=='GK')].copy()
    defender_data=latest_df[(latest_df['Position']=='DF')].copy()
    midfielder_data=latest_df[latest_df['Position']=='MD'].copy()
    forward_data=latest_df[latest_df['Position']=='FW'].copy()

    st.write(goalkeeper_data.set_index('full_name').style.format(format_mapping))
    st.write(defender_data.set_index('full_name').style.format(format_mapping))
    st.write(midfielder_data.set_index('full_name').style.format(format_mapping))
    st.write(forward_data.set_index('full_name').style.format(format_mapping))

    # https://stackoverflow.com/questions/70351068/conditional-formatting-multiple-columns-in-pandas-data-frame-and-saving-as-html

# with st.expander('TEST RUN WORKING THROUGH'):
#     st.write('check this full df',full_df[full_df['full_name'].str.contains('kane')])
#     raw_data = []
#     for n in range(11,14): 
#         # @st.cache(suppress_st_warning=True)
#         def find_latest_player_stats(x):
#             # week_no = st.number_input ("Week number?", min_value=int(1),value=int(20))
#             x=x[x['week'] == n]
#             x = x.sort_values(by=['year', 'week'], ascending=[False, False]).drop_duplicates('full_name')
#             # x = x[x['games_2022_rolling']>2] # want to exclude players who haven't played at all or less than once in 2022 season
#             return x[x['year'] == 2022]

#         latest_df = find_latest_player_stats(full_df)

        
#         latest_df=latest_df.sort_values(by=['last_76_ppg'],ascending=False)
        
#         def ranked_players(x):
#             # only want players who played greater than a season ie 38 games big sample size
#             x = x[x['games_total']>38]
#             x['ppg_76_rank']=x.loc[:,['last_76_ppg']].rank(method='dense', ascending=False)
#             return x

#         def value_rank(x):
#             x['total_selected']=9000000
#             x['%_selected']=x['selected'] / x['total_selected']
#             x['value_ppg']=x['last_76_ppg']/(x['%_selected']+1)
#             # x['value_ppg']=x['last_76_ppg']/(x['%_selected'])
#             x['value_rank']=x.loc[:,['value_ppg']].rank(method='dense', ascending=False)
#             x['selected_rank']=x.loc[:,['%_selected']].rank(method='dense', ascending=False)
#             return x

#         latest_df = ranked_players(latest_df)
#         latest_df = value_rank(latest_df)

#         weekly_transfers_in=read_data('C:/Users/Darragh/Documents/Python/premier_league/week_transfers_in.csv',col_selection=['full_name','transfers_balance'])
#         def merge_latest_transfers(x):
#             return pd.merge(x,weekly_transfers_in,on=['full_name'],how='left')

#         def weekly_transfers_historical(x):
#             x['transfers_balance']=x['transfers_in']-x['transfers_out']
#             return x

#         def rank_calc(x):
#             # USE THIS FOR LATEST TRANSFERS IN WEEK
#             x['net_transfers_rank']=x.loc[:,['transfers_balance']].rank(method='dense', ascending=False)
#             return x

#         def rank_total_calc(x):
#             col_list_1=['ppg_76_rank','value_rank','net_transfers_rank']
#             x['total_sum_rank']=x[col_list_1].sum(axis=1)
#             x['totals_ranked']=x.loc[:,['total_sum_rank']].rank(method='dense', ascending=True)
#             return x

#         latest_df = weekly_transfers_historical(latest_df)
#         latest_df = rank_calc(latest_df)
#         latest_df = rank_total_calc(latest_df)

#         cols_to_move=['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','totals_ranked','total_sum_rank',
#         'ppg_76_rank','value_rank','net_transfers_rank','last_76_ppg','value_ppg','selected_rank','transfers_balance',
#         'last_38_ppg','last_19_ppg','games_total','last_38_games','selected']
#         cols = cols_to_move + [col for col in latest_df if col not in cols_to_move]
#         latest_df=((latest_df[cols].sort_values(by=['totals_ranked'],ascending=True)))
#         # latest_df=latest_df.loc[:['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','totals_ranked','total_sum_rank',
#         # 'ppg_76_rank','value_rank','net_transfers_rank','last_76_ppg','value_ppg','selected_rank']]
#         raw_data.append(latest_df)
#     df1 = pd.concat(raw_data, ignore_index=True)
#     # df1.to_csv('C:/Users/Darragh/Documents/Python/premier_league/gw_analysis_to_date_1.csv')

with st.expander('To run the GW analysis'):
    def run_gw_analysis():
    # st.write('full df',full_df['full_name'])
        raw_data = []
        for n in range(1,future_gameweek): 
            # @st.cache(suppress_st_warning=True)
            def find_latest_player_stats(x):
                # week_no = st.number_input ("Week number?", min_value=int(1),value=int(20))
                x=x[x['week'] == n]
                x = x.sort_values(by=['year', 'week','games_2022_rolling'], ascending=[False, False,False]).drop_duplicates('full_name')
                # x = x[x['games_2022_rolling']>2] # want to exclude players who haven't played at all or less than once in 2022 season
                return x[x['year'] == 2022]

            latest_df = find_latest_player_stats(full_df)
            # st.write('check this latest df', latest_df[latest_df['full_name'].str.contains('bowen')])
            latest_df=latest_df.sort_values(by=['last_76_ppg'],ascending=False)
            
            def ranked_players(x):
                # only want players who played greater than a season ie 38 games big sample size
                x = x[x['games_total']>38]
                # x['ppg_76_rank']=x.loc[:,['last_76_ppg']].rank(method='dense', ascending=False)
                x['ppg_76_rank']=x.loc[:,['last_38_ppg']].rank(method='dense', ascending=False)
                return x

            def value_rank(x):
                x['total_selected']=9000000
                x['%_selected']=x['selected'] / x['total_selected']
                # x['value_ppg']=x['last_76_ppg']/(x['%_selected']+1)
                # x['value_ppg']=x['last_76_ppg']/(x['%_selected']+0.75)
                x['value_ppg']=x['last_76_ppg']/(x['%_selected']+0.5)
                # x['value_ppg']=x['last_76_ppg']/(x['%_selected'])
                x['value_rank']=x.loc[:,['value_ppg']].rank(method='dense', ascending=False)
                x['selected_rank']=x.loc[:,['%_selected']].rank(method='dense', ascending=False)
                return x

            latest_df = ranked_players(latest_df)
            latest_df = value_rank(latest_df)

            weekly_transfers_in=read_data('C:/Users/Darragh/Documents/Python/premier_league/week_transfers_in.csv',col_selection=['full_name','transfers_balance'])
            def merge_latest_transfers(x):
                return pd.merge(x,weekly_transfers_in,on=['full_name'],how='left')

            def weekly_transfers_historical(x):
                x['transfers_balance']=x['transfers_in']-x['transfers_out']
                return x

            def rank_calc(x):
                # USE THIS FOR LATEST TRANSFERS IN WEEK
                x['net_transfers_rank']=x.loc[:,['transfers_balance']].rank(method='dense', ascending=False)
                return x

            def rank_total_calc(x):
                col_list_1=['ppg_76_rank','value_rank','net_transfers_rank']
                x['total_sum_rank']=x[col_list_1].sum(axis=1)
                x['totals_ranked']=x.loc[:,['total_sum_rank']].rank(method='dense', ascending=True)
                return x

            latest_df = weekly_transfers_historical(latest_df)
            latest_df = rank_calc(latest_df)
            latest_df = rank_total_calc(latest_df)

            cols_to_move=['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','totals_ranked','total_sum_rank',
            'ppg_76_rank','value_rank','net_transfers_rank','last_76_ppg','value_ppg','selected_rank','transfers_balance',
            'last_38_ppg','last_19_ppg','games_total','last_38_games','selected']
            cols = cols_to_move + [col for col in latest_df if col not in cols_to_move]
            latest_df=((latest_df[cols].sort_values(by=['totals_ranked'],ascending=True)))
            # latest_df=latest_df.loc[:['full_name','Position','Price','team','week','year','games_2022_rolling','minutes','Clean_Pts','totals_ranked','total_sum_rank',
            # 'ppg_76_rank','value_rank','net_transfers_rank','last_76_ppg','value_ppg','selected_rank']]
            raw_data.append(latest_df)
        df1 = pd.concat(raw_data, ignore_index=True)
        future_week=20
        df1.to_csv('C:/Users/Darragh/Documents/Python/premier_league/gw_analysis_to_date_value.csv')
        return df1
    run_gw_analysis()


with st.expander('Analyse GW data Player Level'):
    # @st.cache
    def load_data(x):
        return pd.read_csv(x)
    data=load_data('C:/Users/Darragh/Documents/Python/premier_league/gw_analysis_to_date_value.csv')
    data_for_processing_current_transfers=data.copy()
    # st.write(data)
    # st.write(data[data['full_name'].str.contains('bowen')])
    # player_detail_data=data.copy().drop('Unnamed: 0',axis=1)
    player_detail_data=data.copy()
    # st.write((player_detail_data[player_detail_data['week']==16]).sort_values(by=['full_name']))

    st.write('Check for duplicates')
    st.write(player_detail_data[player_detail_data.duplicated(subset=['full_name','week'])])
    st.write('Might need to do something for Double GW players')

    player_names_pick=player_detail_data['full_name'].unique()
    names_selected_pick = st.selectbox('Select players',player_names_pick, key='player_pick_data',index=0)
    player_selected_detail_by_week = (player_detail_data[player_detail_data['full_name']==names_selected_pick]).drop('Unnamed: 0',axis=1)
    # st.write(player_selected_detail_by_week.set_index('week').style.format(format_mapping))
    st.write(player_selected_detail_by_week.style.format(format_mapping))
    # st.write(data)
    
with st.expander('Graph GW data'):
    data=data[data['games_2022_rolling']>0]
    st.write(data[data['full_name'].str.contains('bowen')])
    data=data.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    # data=data
    stdc_df=data.loc[:,['Week','Team','cover','Position']].copy()
    merge_historical_stdc_df=stdc_df.copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))

    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    # st.write('check marcus rashford week 16 in 2022 in 3 times??')

    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Graph FW data'):
    data=data[data['games_2022_rolling']>0]
    data=data.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    stdc_df=data.loc[:,['Week','Team','cover','Position']].copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    fw=stdc_df[stdc_df['Position']=='FW'].copy()

    stdc_pivot=pd.pivot_table(fw,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(fw).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    # st.write('check marcus rashford week 16 in 2022 in 3 times??')

    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Graph MD data'):
    data=data[data['games_2022_rolling']>0]
    data=data.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    stdc_df=data.loc[:,['Week','Team','cover','Position']].copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    fw=stdc_df[stdc_df['Position']=='MD'].copy()

    stdc_pivot=pd.pivot_table(fw,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(fw).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    # st.write('check marcus rashford week 16 in 2022 in 3 times??')

    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Graph DF data'):
    data=data[data['games_2022_rolling']>0]
    data=data.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    stdc_df=data.loc[:,['Week','Team','cover','Position']].copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    fw=stdc_df[stdc_df['Position']=='DF'].copy()

    stdc_pivot=pd.pivot_table(fw,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(fw).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    # st.write('check marcus rashford week 16 in 2022 in 3 times??')

    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Graph GK data'):
    data=data[data['games_2022_rolling']>0]
    data=data.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    stdc_df=data.loc[:,['Week','Team','cover','Position']].copy()
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    fw=stdc_df[stdc_df['Position']=='GK'].copy()

    stdc_pivot=pd.pivot_table(fw,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(fw).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    # st.write('check marcus rashford week 16 in 2022 in 3 times??')

    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('GW Detail with Latest Transfers'):
    # current_week=22
    test_data=data_for_processing_current_transfers.drop(['transfers_balance'],axis=1).copy()
    current_data_week=(test_data[test_data['week']==current_week]).copy()
    # st.write(current_data_week.sort_values(by=['Clean_Pts'],ascending=False))
    # current_data_week=current_data_week.copy()

    weekly_transfers_in=read_data('C:/Users/Darragh/Documents/Python/premier_league/week_transfers_in.csv',col_selection=['full_name','transfers_balance'])
    def merge_latest_transfers(x):
        return pd.merge(x,weekly_transfers_in,on=['full_name'],how='left')

    
    def rank_calc(x):
        # USE THIS FOR LATEST TRANSFERS IN WEEK
        x['net_transfers_rank']=x.loc[:,['transfers_balance']].rank(method='dense', ascending=False)
        return x

    def rank_total_calc(x):
        col_list_1=['ppg_76_rank','value_rank','net_transfers_rank']
        x['total_sum_rank']=x[col_list_1].sum(axis=1)
        x['totals_ranked']=x.loc[:,['total_sum_rank']].rank(method='dense', ascending=True)
        return x

    current_data_week=merge_latest_transfers(current_data_week)
    current_data_week=rank_calc(current_data_week)
    current_data_week=rank_total_calc(current_data_week)
    current_data_week['week']=current_week+1
    current_data_week=current_data_week[current_data_week['games_2022_rolling']>0]
    current_week_projections = current_data_week.drop(['Unnamed: 0'],axis=1).set_index('full_name').sort_values(by=['totals_ranked'],ascending=True) 
    st.write('Projections', current_week_projections.style.format(format_mapping))
    
with st.expander('GW Graph with Latest Transfers'):    
    current_data_week=current_data_week.rename(columns={'totals_ranked':'cover','full_name':'Team','week':'Week'})
    current_data_week_df=current_data_week.loc[:,['Week','Team','cover','Position']].copy()
    stdc_df=pd.concat([merge_historical_stdc_df,current_data_week_df])
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    my_players_data=stdc_df.copy()
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('GW Graph with My Players'):
    my_players=['harry_kane','marcos_alonso','bruno miguel_borges fernandes','trent_alexander-arnold',
    'michail_antonio','joão pedro cavaco_cancelo','mason_mount','jarrod_bowen','diogo_jota']

    # st.write(my_players_data)
    stdc_df=my_players_data[my_players_data['Team'].isin(my_players)]
    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('white'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)