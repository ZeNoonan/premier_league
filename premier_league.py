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

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

with st.beta_expander('Mins'):

    # @st.cache
    def prep_base_data(url_csv, pick):
        url_csv = pd.read_csv(url_csv).rename(columns = {'id':'player_id','element_type':'Position','assists':'assists_season',
        'bonus':'bonus_season','bps':'bps_season','clean_sheets':'clean_sheets_season','creativity':'creativity_season','goals_conceded':'goals_conceded_season',
        'goals_scored':'goals_scored_season','ict_index':'ict_index_season','influence':'influence_season','minutes':'minutes_season',
        'saves':'saves_season','threat':'threat_season','transfers_in':'transfers_in_season','transfers_out':'transfers_out_season'})
        url_csv['Position'] = url_csv['Position'].map({1: 'GK', 2: 'DF', 3:'MD', 4:'FW'})
        url_csv['full_name'] = (url_csv['first_name']+'_'+url_csv['second_name']).str.lower()
        pick_data = pick.rename(columns = {'total_points':'week_points'})
        return pd.merge(url_csv,pick_data, on='player_id',how ='outer')

    # @st.cache(suppress_st_warning=True)
    def data_2022_team_names(file):
        file['team'] = file['team'].map({1: 'Arsenal', 2: 'Aston_Villa', 3: 'Brentford', 4:'Brighton', 5:'Burnley',6:'Chelsea',7:'Crystal_Palace',8:'Everton',
        9:'Leicester',10:'Leeds_Utd',11:'Liverpool',12:'Man_City',13:'Man_Utd',14:'Newcastle',15:'Norwich',16:'Southampton',17:'Spurs',
        18:'Watford',19:'West_Ham',20:'Wolves'})
        return file

    url2022 = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/players_raw.csv'
    df_week_data_raw=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/raw_data_2022.csv')
    odds_data=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/goalscorer_odds.xlsx')

    data_2022 = (data_2022_team_names( (prep_base_data(url2022, df_week_data_raw)).rename(columns = {'team_x':'team'})))

    # @st.cache(suppress_st_warning=True)
    def column_calcs(df):
        df = df.sort_values(by=['full_name', 'year', 'week'], ascending=[True, True, True])
        df['Price'] =df['value'] / 10
        df['Game_1'] = np.where((df['minutes'] > 0.5), 1, 0)
        df['Clean_Pts'] = np.where(df['Game_1']==1,df['week_points'], np.NaN) # setting a slice on a slice - just suppresses warning....
        df_calc=df[df['Game_1']>0].copy()
        df_calc['4_games_rolling_mins']=df_calc.groupby(['full_name'])['minutes'].rolling(window=4,min_periods=1, center=False).sum().reset_index(0,drop=True)
        df=pd.merge(df,df_calc,how='outer').sort_values(by=['full_name', 'year', 'week'], ascending=[True, True, True])
        cols_to_move = ['full_name','week','year','Price' ,'minutes','Clean_Pts','Game_1','week_points','4_games_rolling_mins','team']
        cols = cols_to_move + [col for col in df if col not in cols_to_move]
        df=df[cols]
        df['4_games_rolling_mins']=df.groupby('full_name')['4_games_rolling_mins'].ffill()
        return df

    data_2022=column_calcs(data_2022).copy()
    # st.write('data 2022', data_2022)
    cols_to_move = ['full_name','week','year','Price' ,'minutes','Clean_Pts','Game_1','week_points','4_games_rolling_mins']
    cols = cols_to_move + [col for col in data_2022 if col not in cols_to_move]
    data_2022=data_2022[cols]

    player_names_pick=data_2022['full_name'].unique()
    names_selected_pick = st.selectbox('Select players',player_names_pick, key='player_pick',index=0)
    player_selected_detail_by_week = data_2022[data_2022['full_name']==names_selected_pick]
    st.write('what week is used here')
    st.write( player_selected_detail_by_week.sort_values(by=['year','week'],ascending=[False,False]) )

    week_mins = 10
    current_week=11
    df_1= data_2022 [ (data_2022['week']==week_mins) ].sort_values(by='Price',ascending=False)
    df_1=df_1.loc[:,['full_name','week','year','Price','4_games_rolling_mins','team']]
    df_1['week']=week_mins+1
    # st.write('mins fantasy check', df_1)
    # st.write('odds check week',odds_data)
    odds_data=odds_data[odds_data['week']==current_week]
    # st.write('check after',odds_data)
    df_1=pd.merge(df_1,odds_data,on=['full_name','team','week'],how='outer')
    # st.write('checking df1', df_1)
    # get rid of any blanks in odds data so that ranking is not upset
    df_1=df_1.dropna(subset=['odds_betfair'])
    df_1['odds_betfair_rank']=df_1['odds_betfair'].rank(method='dense', ascending=True)
    df_1['nan_pinnacle']=np.where(df_1['odds_pinnacle']>0.1,1,np.NaN)
    # st.write('checking nan', df_1)
    df_1['odds_pinnacle_rank']=df_1['odds_pinnacle'].rank(method='dense', ascending=True)
    df_1['rolling_mins_rank']=df_1['4_games_rolling_mins'].rank(method='dense', ascending=False)
    # st.write('data',df_1.sort_values(by='odds_betfair',ascending=True))
    
    
    def team_names(file):
        file['team'] = file['team'].map({'Arsenal': 'Arsenal', 'Aston Villa': 'Aston_Villa', 'Brentford': 'Brentford', 'Brighton':'Brighton',
         'Burnley':'Burnley','Chelsea':'Chelsea','Crystal Palace':'Crystal_Palace','Everton':'Everton',
        'Leicester City':'Leicester','Leeds United':'Leeds_Utd','Liverpool':'Liverpool','Manchester City':'Man_City',
        'Manchester Utd':'Man_Utd','Newcastle Utd':'Newcastle','Norwich City':'Norwich','Southampton':'Southampton','Tottenham':'Spurs',
        'Watford':'Watford','West Ham':'West_Ham','Wolves':'Wolves'})
        return file

    team_xg = team_names(pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/team_xg.csv'))
    # team_xg = (pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/team_xg.csv'))
    # st.write('team xg', team_xg)
    # st.markdown(get_table_download_link(df_1), unsafe_allow_html=True)
    

with st.beta_expander('df'):
    # dfa=pd.read_html('https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures')
    # dfa[0].to_pickle('C:/Users/Darragh/Documents/Python/premier_league/scores.pkl')
    df=pd.read_pickle('C:/Users/Darragh/Documents/Python/premier_league/scores.pkl')
    df=df.dropna(subset=['Wk'])
    # st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    odds = pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx')

    merged_df = pd.merge(df,odds,on=['Wk','Day','Date','Home','xG','Score','xG.1','Away'],how='outer')
    # https://stackoverflow.com/questions/35552874/get-first-letter-of-a-string-from-column
    merged_df['Home Points'] = [str(x)[0] for x in merged_df['Score']]
    merged_df['Away Points'] = [str(x)[2] for x in merged_df['Score']]
    merged_df['home_spread']=-merged_df['Spread']
    merged_df['away_spread']=merged_df['Spread']
    # merged_df=team_names(merged_df)
    home_spread=merged_df.loc[:,['Wk','Home','home_spread']].rename(columns={'Wk':'week','Home':'team','home_spread':'spread'})
    # st.write('home spread', home_spread)
    away_spread=merged_df.loc[:,['Wk','Away','away_spread']].rename(columns={'Wk':'week','Away':'team','away_spread':'spread'})
    combined_spread = pd.concat([home_spread,away_spread],axis=0)
    combined_spread=team_names(combined_spread)
    # st.write('combined spread???', combined_spread.sort_values(by='week',ascending=True))
    # st.write('to be merged with df1', df_1)
    # df_1['week']=df_1['week']+1
    df_update = pd.merge(df_1, combined_spread,on=['week','team'], how='left')
    df_update['spread_rank']=df_update['spread'].rank(method='dense', ascending=False)
    col_list=['spread_rank','odds_betfair_rank','rolling_mins_rank']
    col_list_1=['spread_rank','odds_pinnacle_rank','rolling_mins_rank']
    df_update['total_betfair_rank']=df_update[col_list].sum(axis=1)
    df_update['total_pinnacle_rank']=df_update[col_list_1].sum(axis=1)*df_update['nan_pinnacle']
    # st.write(df_update)
    df_update['factor_betfair_rank']=df_update['total_betfair_rank'].rank(method='dense', ascending=True)
    df_update['factor_pinnacle_rank']=df_update['total_pinnacle_rank'].rank(method='dense', ascending=True)
    # df_update = pd.merge(df_update, away_spread,on=['week','team'], how='left')
    df_update['log_rank']=np.log(df_update['spread_rank'])
    cols_to_move = ['full_name','week','spread','team','spread_rank','odds_pinnacle_rank','rolling_mins_rank','factor_pinnacle_rank','factor_betfair_rank','year','Price' ,'4_games_rolling_mins']
    cols = cols_to_move + [col for col in df_update if col not in cols_to_move]
    df_update=df_update[cols].sort_values(by='factor_pinnacle_rank').reset_index().drop('index',axis=1)
    st.write('merged', df_update)

    # goal_df = pd.read_html('https://www.pinnacle.com/en/soccer/england-premier-league/watford-vs-southampton/1420054427#player-props')
    # st.write(goal_df)
    # team_names_id=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx',sheet_name='Sheet2')

    # st.write(team_names_id)
    # st.write('merged', merged_df)

    # st.write(odds)
    # st.write(df)



