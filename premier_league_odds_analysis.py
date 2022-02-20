import pandas as pd
import numpy as np
from pandas.core.reshape.merge import merge
import streamlit as st
from io import BytesIO
import os
import base64 
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import seaborn as sns

st.set_page_config(layout="wide")
current_week=25
finished_week=25

placeholder_1=st.empty()
placeholder_2=st.empty()

with st.expander('df'):
    # dfa=pd.read_html('https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures')
    # dfa[0].to_pickle('C:/Users/Darragh/Documents/Python/premier_league/scores.pkl')
    # dfa[0].to_csv('C:/Users/Darragh/Documents/Python/premier_league/scores.csv')
    df=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/scores.csv',parse_dates=['Date'])
    # st.write(df)

    df=df.dropna(subset=['Wk'])
    # st.write('duplicates in df??', df)
    def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    # csv = convert_df(df)
    # st.download_button(label="Download data as CSV",data=csv,file_name='df.csv',mime='text/csv',key='scores')


    # st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    odds = pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx',parse_dates=['Date'])
    prior_data=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/prior_year.xlsx',parse_dates=['Date'])
    # odds = pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league_dummy.xlsx',parse_dates=['Date'])
    # prior_data=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/prior_year_dummy.xlsx',parse_dates=['Date'])

    def concat_current_prior(x,y):
        current_plus_prior = pd.concat([x,y],axis=0,ignore_index=True)
        return current_plus_prior

    odds=concat_current_prior(odds,prior_data).drop(['xG','Score','xG.1'],axis=1)
    # st.write('odds duplicates???????', odds)
    # st.write('df',df)
    # st.write('Date type in odds',odds['Date'].dtype)
    # st.write('Date type in df',df['Date'].dtype)
    merged_df = pd.merge(df,odds,on=['Date','Home','Away'],how='outer').drop(['Day_y','Wk_x'],axis=1)\
        .rename(columns={'xG_x':'xG','xG.1_x':'xG.1','Score_x':'Score','Day_x':'Day','Date_x':'Date','Home':'Home Team','Away':'Away Team','Wk_y':'Week'})
    # st.write('merged df',merged_df[merged_df['Spread'].isna()])
    # st.write('DUPLICATES IN HERE?? merged df',merged_df)
    # https://stackoverflow.com/questions/35552874/get-first-letter-of-a-string-from-column
    merged_df['Home Points'] = [str(x)[0] for x in merged_df['Score']]
    merged_df['Away Points'] = [str(x)[2] for x in merged_df['Score']]
    merged_df['home_spread']=merged_df['Spread']
    merged_df['away_spread']=-merged_df['Spread']
    merged_df=merged_df[merged_df['Notes']!='Match Postponed']
    merged_df['Home Points']=merged_df['Home Points'].replace({'n':np.NaN})
    merged_df['Away Points']=merged_df['Away Points'].replace({'n':np.NaN})
    merged_df['Home Points']=pd.to_numeric(merged_df['Home Points'])
    merged_df['Away Points']=pd.to_numeric(merged_df['Away Points'])
    # st.write(merged_df.dtypes)
    data=merged_df
    # csv = convert_df(data)
    # st.download_button(label="Download data as CSV",data=csv,file_name='df.csv',mime='text/csv',key='after_merge')
    # st.write(data)

    def spread_workings(data):
        data['home_win']=data['Home Points'] - data['Away Points']
        data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
        
        data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
        np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
        
        data['first_spread']=(np.where(((data['Home Points'] + (data['Spread']+0.25)) > data['Away Points']), 1,
        np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))*.5
        data['second_spread']=(np.where(((data['Home Points'] + (data['Spread']-0.25)) > data['Away Points']), 1,
        np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))*.5
        data['home_cover']=data['first_spread']+data['second_spread']

        # data['home_cover']=data['home_cover'].astype(int)
        data['home_cover']=data['home_cover'].astype(float)
        data['away_cover'] = -data['home_cover']
        return data

    # def spread_workings_new(data):
    #     data['first_spread']=(np.where(((data['Home Points'] + (data['Spread']+0.25)) > data['Away Points']), 1,
    #     np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))*.5
    #     data['second_spread']=(np.where(((data['Home Points'] + (data['Spread']-0.25)) > data['Away Points']), 1,
    #     np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))*.5
    #     data['home_cover']=data['first_spread']+data['second_spread']
    #     return data

    # st.write('is there duplicates in here', data)
    spread=spread_workings(data)
    st.write('is home cover working right', spread)
    workings_1=spread.copy()

    # test_spread=spread_workings_new(data)
    # st.write('test spread',spread)


    team_names_id=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx', sheet_name='Sheet2')
    # st.write(team_names_id)
    team_names_id=team_names_id.rename(columns={'team':'Home Team'})
    # st.write('this is spread before merge', spread)
    # st.write('Date dtype', spread['Date'].dtype)
    # st.write('this is team names id before merge', team_names_id)
    odds_data=pd.merge(spread,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'}).sort_values(by='Date',ascending=False)
    team_names_id=team_names_id.rename(columns={'Home Team':'Away Team'})
    odds_data=pd.merge(odds_data,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'}).sort_values(by='Date',ascending=False)
    # st.write(odds_data)

    matrix_df=odds_data.reset_index().rename(columns={'index':'unique_match_id'})
    workings_1=matrix_df.copy()

    # st.write('matrix df',matrix_df)
    test_df = matrix_df.copy()
    # st.write('check for unique match id', test_df)
    matrix_df['at_home'] = 1
    matrix_df['at_away'] = -1

    matrix_df['home_pts_adv'] = -0.25
    matrix_df['away_pts_adv'] = 0.25


    test_df_1=matrix_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
    test_df_home=test_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
    test_df_away=test_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
    test_df_2=pd.concat([test_df_home,test_df_away],ignore_index=True)
    test_df_2=test_df_2.sort_values(by=['ID','Week'],ascending=True)
    test_df_2['spread_with_home_adv']=test_df_2['spread']+test_df_2['home_pts_adv']
    # st.write(test_df_2)

    first_qtr=matrix_df.copy()
    start=-3
    finish=0
    first_4=first_qtr[first_qtr['Week'].between(start,finish)].copy()
    def games_matrix_workings(first_4):
        group_week = first_4.groupby('Week')
        raw_data_2=[]
        game_weights = iter([-0.125, -0.25,-0.5,-1])
        for name, group in group_week:
            group['game_adj']=next(game_weights)
            # st.write('looking at for loop',group)
            raw_data_2.append(group)

        df3 = pd.concat(raw_data_2, ignore_index=True)
        adj_df3=df3.loc[:,['Home ID', 'Away ID', 'game_adj']].copy()
        test_adj_df3 = adj_df3.rename(columns={'Home ID':'Away ID', 'Away ID':'Home ID'})
        concat_df_test=pd.concat([adj_df3,test_adj_df3]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        test_concat_df_test=concat_df_test.groupby('Home ID')['game_adj'].sum().abs().reset_index()
        test_concat_df_test['Away ID']=test_concat_df_test['Home ID']
        full=pd.concat([concat_df_test,test_concat_df_test]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        full_stack=pd.pivot_table(full,index='Away ID', columns='Home ID',aggfunc='sum')
        # st.write('Check sum looks good all zero', full_stack.sum())
        full_stack=full_stack.fillna(0)
        full_stack.columns = full_stack.columns.droplevel(0)
        return full_stack
    full_stack=games_matrix_workings(first_4)
    # st.write(full_stack)

def test_4(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['adj_spread']=matrix_df_1['spread_with_home_adv'].rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    return matrix_df_1


# with st.beta_expander('CORRECT Power Ranking to be used in Matrix Multiplication'):
# # https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
grouped = test_df_2.groupby('ID')
# https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence
# https://stackoverflow.com/questions/62471485/is-it-possible-to-insert-missing-sequence-numbers-in-python
ranking_power=[]
for name, group in grouped:
    dfseq = pd.DataFrame.from_dict({'Week': range( -3,38 )}).merge(group, on='Week', how='outer').fillna(np.NaN)
    dfseq['ID']=dfseq['ID'].fillna(method='ffill')
    dfseq['home_pts_adv']=dfseq['home_pts_adv'].fillna(0)
    dfseq['spread']=dfseq['spread'].fillna(0)
    dfseq['spread_with_home_adv']=dfseq['spread_with_home_adv'].fillna(0)
    dfseq['home']=dfseq['home'].fillna(0)
    df_seq_1 = dfseq.groupby(['Week','ID'])['spread_with_home_adv'].sum().reset_index()
    update=test_4(df_seq_1)
    ranking_power.append(update)
df_power = pd.concat(ranking_power, ignore_index=True)

inverse_matrix=[]
power_ranking=[]
list_inverse_matrix=[]
list_power_ranking=[]
power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()

games_df=matrix_df.copy()
# st.write('Checking the games df', games_df[((games_df['Home ID']==24)|(games_df['Away ID']==24))])
first=list(range(-3,35))
last=list(range(0,38))
for first,last in zip(first,last):
    first_section=games_df[games_df['Week'].between(first,last)]
    full_game_matrix=games_matrix_workings(first_section)
    adjusted_matrix=full_game_matrix.loc[0:18,0:18]
    df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
    power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
    .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:18,:]
    result = df_inv.dot(pd.DataFrame(power_df_week))
    result.columns=['power']
    avg=(result['power'].sum())/20
    result['avg_pwr_rank']=(result['power'].sum())/20
    result['final_power']=result['avg_pwr_rank']-result['power']
    df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
    result=pd.concat([result,df_pwr],ignore_index=True)
    result['week']=last+1
    power_ranking.append(result)
power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})

matches_df = matrix_df.copy()
home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
np.where(updated_df['spread_working'] < 0,-1,0))

def season_cover_workings(data,home,away,name,week_start):
    season_cover_df=data[data['Week']>week_start].copy()
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Date','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    # st.write('checking home turnover section', home_cover_df[home_cover_df['ID']==0])
    away_cover_df = (season_cover_df.loc[:,['Week','Date','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    # st.write('checking away turnover section', away_cover_df[away_cover_df['ID']==0])
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=[True,True,True])

def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    # season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    # THE ABOVE DIDN'T WORK IN 2020 PRO FOOTBALL BUT DID WORK IN 2019 DO NOT DELETE FOR INFO PURPOSES
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

# st.write('this is the base data....check', matrix_df)
spread_1 = season_cover_workings(matrix_df,'home_cover','away_cover','cover',0)
spread_2=season_cover_2(spread_1,'cover')
# st.write('spread 2 should this not be adding up to 0.5 increments for some of them', spread_2)
spread_3=season_cover_3(spread_2,'cover_sign','cover')
# st.write('spread 3', spread_3)


# turnover=spread_workings(data)



# st.write('turnover workings', turnover)


with st.expander('Turnover equival workings xg'):
    # workings_1=spread.copy()
    # st.write('workings any dupicates in here??',workings_1)
    workings_1['xg_margin']=workings_1['xG']-workings_1['xG.1']
    workings_1['home_xg_win']=np.where(workings_1['xg_margin']>1.0,1,np.where(workings_1['xg_margin']<-1.0,-1,0))
    workings_1=workings_1.assign(result_str=workings_1['home_xg_win'].astype(str)+workings_1['home_win'].astype(str))
    # st.write(workings_1)
    workings_1['Turnover'] = workings_1['result_str'].replace({'00':0,'01':-1,'10':1,'0-1':1,'11':0,'-1-1':0,'-10':-1,'1-1':1,'-11':-1})
    cols_to_move=['Week','Date','Home ID','Home Team','xG','Score','xG.1','Away Team','Away ID','xg_margin','home_xg_win','home_win','result_str','Turnover']
    cols = cols_to_move + [col for col in workings_1 if col not in cols_to_move]
    turnover=workings_1[cols]
    # st.write('turnover use this for sense checking',turnover[(turnover['Home ID']==0) | (turnover['Away ID']==0]))
    st.write( turnover[(turnover['Home ID']==0) | (turnover['Away ID']==0)] )

    def turnover_workings(data,week_start):
        turnover_df=data[data['Week']>week_start].copy()
        turnover_df['home_turned_over_sign'] = np.where((turnover_df['Turnover'] > 0), 1, np.where((turnover_df['Turnover'] < 0),-1,0))
        turnover_df['away_turned_over_sign'] = - turnover_df['home_turned_over_sign']
        # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
        home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_turned_over_sign']]).rename(columns={'Home ID':'ID','home_turned_over_sign':'turned_over_sign'})
        # st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
        away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_turned_over_sign']]).rename(columns={'Away ID':'ID','away_turned_over_sign':'turned_over_sign'})
        # st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
        season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
        # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
        # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
        return season_cover.sort_values(by=['Week','Date','ID'],ascending=[True,True,True])

    def turnover_2(season_cover_df):    
        # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
        season_cover_df['prev_turnover']=season_cover_df.groupby('ID')['turned_over_sign'].shift()
        return season_cover_df.sort_values(by=['ID','Week'],ascending=True)
        # return season_cover_df



    turnover_1 = turnover_workings(turnover,-1)
    turnover_2=turnover_2(turnover_1)
    turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')

    def turnover_data_prep_1(turnover_3):
        return turnover_3.loc[:,['Date','Week','ID','prev_turnover', 'turnover_sign']].copy()

    turnover_matches = turnover_data_prep_1(turnover_3)
    # st.write('turnover matches', turnover_matches)

    def turnover_data_prep_2(turnover_matches,updated_df):
        turnover_home=turnover_matches.rename(columns={'ID':'Home ID'})
        turnover_away=turnover_matches.rename(columns={'ID':'Away ID'})
        turnover_away['turnover_sign']=-turnover_away['turnover_sign']
        updated_df=pd.merge(updated_df,turnover_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_turnover':'home_prev_turnover','turnover_sign':'home_turnover_sign'})
        updated_df=pd.merge(updated_df,turnover_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_turnover':'away_prev_turnover','turnover_sign':'away_turnover_sign'})
        return updated_df

    updated_df = turnover_data_prep_2(turnover_matches, updated_df)

    df_stdc_1=pd.merge(turnover_matches,team_names_id,on='ID').rename(columns={'Away Team':'Team'})
    # st.write('issue with Team?',df_stdc_1)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['turnover_sign'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["red", "lightgrey","LimeGreen"])

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=color_scale))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    
    text_cover=chart_cover.mark_text().encode(text=alt.Text('turnover_sign:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)



with st.expander('Season to Date Cover Graph'):
    st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    st.write('blanks in graph are where the team got a bye week')
    # df = pd.DataFrame([['mon',19,'cardinals', 3], ['tue',20,'patriots', 4], ['wed',20,'patriots', 5]], columns=['date','week','team', 'stdc'])
    # st.write('df1',df)
    # df2 = pd.DataFrame([['sun',18,'saints'], ['tue',20,'patriots'], ['wed',20,'patriots']], columns=['date','week','team'])
    # st.write('df2',df2)
    # df3=df2.merge(df,on=['date','week','team'], how='left')
    # st.write('merged on left',df3)  # merges on columns A

    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=updated_df.drop(['away_cover'],axis=1)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign':'away_cover_sign'})
    # st.write('check that STDC coming in correctly', updated_df)
    # st.write('Check Total')
    # st.write('home',updated_df['home_cover_sign'].sum())
    # st.write('away',updated_df['away_cover_sign'].sum())
    # st.write('Updated for STDC', updated_df)
    # st.write('Get STDC by Week do something similar for Power Rank')
    # last_occurence = spread_3.groupby(['ID'],as_index=False).last()
    # st.write(last_occurence)
    stdc_df=pd.merge(spread_3,team_names_id,on='ID').rename(columns={'Away Team':'Team'})
    team_names_id_update=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    df_stdc_1=pd.merge(spread_3,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1=df_stdc_1.loc[:,['Week','ID','Team','cover']].copy()
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N',format=",.1f"),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    team_names_id=team_names_id.rename(columns={'Away Team':'Team'})
    id_names=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    pivot_df=pd.merge(power_week,id_names, on='ID')
    # st.write('after merge', pivot_df)
    pivot_df=pivot_df.loc[:,['Team','final_power','week']].copy()
    # st.write('graphing?',pivot_df)
    power_pivot=pd.pivot_table(pivot_df,index='Team', columns='week')
    pivot_df_test = pivot_df.copy()
    pivot_df_test=pivot_df_test[pivot_df_test['week']<current_week+2]
    pivot_df_test['average']=pivot_df.groupby('Team')['final_power'].transform(np.mean)
    # st.write('graphing?',pivot_df_test)
    power_pivot.columns = power_pivot.columns.droplevel(0)
    power_pivot['average'] = power_pivot.mean(axis=1)
    # st.write(power_pivot)
    # https://stackoverflow.com/questions/67045668/altair-text-over-a-heatmap-in-a-script
    pivot_df=pivot_df.sort_values(by='final_power',ascending=False)
    chart_power= alt.Chart(pivot_df_test).mark_rect().encode(alt.X('week:O',axis=alt.Axis(title='week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('final_power:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text=chart_power.mark_text().encode(text=alt.Text('final_power:N',format=",.1f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
st.write('updated df', updated_df)
with st.expander('Betting Slip Matches'):
    def run_analysis(updated_df):
        betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
        'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign',
        'home_cover_sign','away_cover_sign','power_pick','home_cover_result']]
        betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
        betting_matches['away_cover_sign']+betting_matches['power_pick']
        betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
        betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
        betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
        betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
        betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
        st.write('testing sum of betting result',betting_matches['result'].sum())
        # this is for graphing anlaysis on spreadsheet
        betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
        betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
        st.write('testing sum of betting all result',betting_matches['result_all'].sum())
        cols_to_move=['Week','Date','Home Team','Away Team','total_factor','bet_on','bet_sign','home_cover_result','result','Spread','Home Points','Away Points',
        'home_cover','away_cover']
        cols = cols_to_move + [col for col in betting_matches if col not in cols_to_move]
        betting_matches=betting_matches[cols]
        betting_matches=betting_matches.sort_values(['Week','Date'],ascending=[True,True])
        return betting_matches
    # st.write(betting_matches.dtypes)
    # st.write(betting_matches)
    # betting_matches_penalty=run_analysis(penalty_df)
    # betting_matches_intercept=run_analysis(updated_df_intercept)
    # betting_matches_sin_bin=run_analysis(updated_df_sin_bin)
    betting_matches=run_analysis(updated_df)
    presentation_betting_matches=betting_matches.copy()

    # https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
    grid_height = st.number_input("Grid height", min_value=400, value=550, step=100)
    gb = GridOptionsBuilder.from_dataframe(presentation_betting_matches)
    gb.configure_column("Spread", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("home_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("away_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("Date", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='dd-MM-yyyy', pivot=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)



    test_cellsytle_jscode = JsCode("""
    function(params) {
        if (params.value < 0) {
        return {
            'color': 'red',
        }
        } else {
            return {
                'color': 'black',
            }
        }
    };
    """)
    # # https://github.com/PablocFonseca/streamlit-aggrid/blob/main/st_aggrid/grid_options_builder.py
    gb.configure_column(field="Spread", cellStyle=test_cellsytle_jscode)
    gb.configure_column("home_power", cellStyle=test_cellsytle_jscode)
    gb.configure_column("away_power", cellStyle=test_cellsytle_jscode)


    # gb.configure_pagination()
    # gb.configure_side_bar()
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    grid_response = AgGrid(
        presentation_betting_matches, 
        gridOptions=gridOptions,
        height=grid_height, 
        width='100%',
        # data_return_mode=return_mode_value, 
        # update_mode=update_mode_value,
        # fit_columns_on_grid_load=fit_columns_on_grid_load,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=True,
    )

    # container.grid_response
    # AgGrid(betting_matches.sort_values('Date').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}"}))
    # update




    st.write('Below is just checking an individual team')
    betting_matches_team=betting_matches.copy()
    # cols_to_move_now=['Week','Home Team','Away Team','Spread','Home Points','Away Points','home_cover_result','home_cover','away_cover']
    # cols = cols_to_move_now + [col for col in betting_matches_team if col not in cols_to_move_now]
    # betting_matches_team=betting_matches_team[cols]
    # st.write( betting_matches_team[(betting_matches_team['Home Team']=='Melbourne Storm') | 
    # (betting_matches_team['Away Team']=='Melbourne Storm')].sort_values(by='Week').set_index('Week') )

with st.expander('Power Pick Factor by Team'):
    st.write('Positive number means the market has undervalued the team as compared to the spread')
    st.write('Negative number means the market has overvalued the team as compared to the spread')    
    power_factor=betting_matches.loc[:,['Week','Home Team','Away Team','power_pick']].rename(columns={'power_pick':'home_power_pick'})
    power_factor['away_power_pick']=-power_factor['home_power_pick']
    home_factor=power_factor.loc[:,['Week','Home Team','home_power_pick']].rename(columns={'Home Team':'Team','home_power_pick':'power_pick'})
    away_factor=power_factor.loc[:,['Week','Away Team','away_power_pick']].rename(columns={'Away Team':'Team','away_power_pick':'power_pick'})
    graph_power_pick=pd.concat([home_factor,away_factor],axis=0).sort_values(by=['Week'])
    graph_power_pick['average']=graph_power_pick.groupby('Team')['power_pick'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["LimeGreen", "lightgrey","red"])

    chart_cover= alt.Chart(graph_power_pick).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('power_pick:Q',scale=color_scale))
    text_cover=chart_cover.mark_text().encode(text=alt.Text('power_pick:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)
    # st.write('graph',graph_power_pick)
    # st.write('data',power_factor)



with st.expander('Analysis of Betting Results across 1 to 5 factors'):
    # matches_in_regular_season= (32 * 16) / 2
    # st.write('In 2020 there were 13 matches in playoffs looks like this was new so 269 total matches in 2020 season compared with 267 in previous seasons')
    # matches_in_playoffs = 13
    # total_matches =matches_in_regular_season + matches_in_playoffs
    # st.write('total_matches per my calculation',total_matches)
    analysis=betting_matches.copy()
    analysis=analysis[analysis['Week']<finished_week+1]
    # st.write('analysis',analysis)
    totals = analysis.groupby('total_factor').agg(winning=('result_all','count'))
    totals_graph=totals.reset_index().rename(columns={'winning':'number_of_games'})

    # st.write('totals grpah', totals_graph)
    chart_power= alt.Chart(totals_graph).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='total_factor_per_match',labelAngle=0)),
    alt.Y('number_of_games'))
    text=chart_power.mark_text(dy=-7).encode(text=alt.Text('number_of_games:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    
    

    totals_1=analysis.groupby([analysis['total_factor'].abs(),'result_all']).agg(winning=('result_all','count')).reset_index()
    totals_1['result_all']=totals_1['result_all'].replace({0:'tie',1:'win',-1:'lose'})
    totals_1['result_all']=totals_1['result_all'].astype(str)
    # st.write(totals_1['result_all'].dtypes)
    # st.write(totals_1['winning'].dtypes)
    # st.write(totals_1['total_factor'].dtypes)
    # st.write('checking graph data',totals_1.dtypes)
    # totals_1['total_factor']=totals_1['total_factor'].astype(str)
    # st.write('checking graph data',totals_1.dtypes)
    # st.write('checking graph data',totals_1)
    # https://www.quackit.com/css/css_color_codes.cfm
    color_scale = alt.Scale(
    domain=[
        "lose",
        "tie",
        "win"],
        range=["red", "lightgrey","LimeGreen"])
    chart_power= alt.Chart(totals_1).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning'),color=alt.Color('result_all',scale=color_scale))
    # st.altair_chart(chart_power,use_container_width=True)

    
    normalized_table = (totals_1[totals_1['result_all']!='tie']).copy()
    # st.write('graph date to be cleaned',totals_1)
    chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning',stack="normalize"),color=alt.Color('result_all',scale=color_scale))
    overlay = pd.DataFrame({'winning': [0.5]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+60, color='white').encode(
    x=alt.X('total_factor:O'),
    y=alt.Y('winning',stack="normalize"),
    detail='winning',
    text=alt.Text('winning:Q', format='.0f'))
    updated_test_chart=chart_power+vline+text
    
    st.altair_chart(updated_test_chart,use_container_width=True)

    # st.write('shows the number of games at each factor level')
    # st.write(totals.rename(columns={'winning':'number_of_games'}))
    # st.write('sum of each factor level should correspond to table above',totals_1)
    # st.write('sum of winning column should be 267 I think',totals_1['winning'].sum())
    # st.write('count of week column should be 267',analysis['Week'].count())

    reset_data=totals_1.copy()
    reset_data['result_all']=reset_data['result_all'].replace({'tie':0,'win':1,'lose':-1})
    # st.write('test',reset_data)
    reset_data=reset_data.pivot(index='result_all',columns='total_factor',values='winning').fillna(0)
    # st.write('look',reset_data)
    reset_data['betting_factor_total']=reset_data[3]+reset_data[4]+reset_data[5]
    reset_data=reset_data.sort_values(by='betting_factor_total',ascending=False)

    reset_data=reset_data.reset_index()
    # st.write('reset data', reset_data)
    reset_data['result_all']=reset_data['result_all'].astype(str)
    reset_data=reset_data.set_index('result_all')

    reset_data.loc['Total']=reset_data.sum()
    reset_data.loc['No. of Bets Made'] = reset_data.loc[['1','-1']].sum() 
    reset_data=reset_data.apply(pd.to_numeric, downcast='integer')
    reset_data.loc['% Winning'] = ((reset_data.loc['1'] / reset_data.loc['No. of Bets Made'])).replace({'<NA>':np.NaN})
    st.write('This shows the betting result')
    # st.write(reset_data)

    # https://stackoverflow.com/questions/64428836/use-pandas-style-to-format-index-rows-of-dataframe
    reset_data = reset_data.style.format("{:.0f}", na_rep='-')
    reset_data = reset_data.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(reset_data)
    st.write('Broken down by the number of factors indicating the strength of the signal')

with placeholder_1.expander('Weekly Results'):
    weekly_results=analysis.groupby(['Week','result']).agg(winning=('result','sum'),count=('result','count'))
    weekly_test=analysis[analysis['total_factor'].abs()>2].loc[:,['Week','result']].copy()

    df9 = weekly_test.groupby(['result','Week']).size().unstack(fill_value=0)
    df9=df9.reset_index()
    df9['result']=df9['result'].astype(int).astype(str)
    df9=df9.set_index('result').sort_index(ascending=False)
    # df9.columns = df9.columns.astype(str)
    df_slice=df9.loc[:,13:]
    df9['subtotal_week_13_on']=df_slice.sum(axis=1)
    df_all=df9.iloc[:,:-1]
    # st.write('this is df all',df_all)
    df9['grand_total']=df_all.sum(axis=1)
    df9.loc['Total']=df9.sum()
    df9.loc['No. of Bets Made'] = df9.loc[['1','-1']].sum() 
    df9=df9.apply(pd.to_numeric, downcast='integer')
    df9.loc['% Winning'] = ((df9.loc['1'] / df9.loc['No. of Bets Made'])).replace({'<NA>':np.NaN})

    # https://stackoverflow.com/questions/64428836/use-pandas-style-to-format-index-rows-of-dataframe
    df9 = df9.style.format("{:.0f}", na_rep='-')
    df9 = df9.format(formatter="{:.0%}", subset=pd.IndexSlice[['% Winning'], :])
    st.write(df9)

    st.write('Total betting result',betting_matches['result'].sum())
    # pivot_weekly.columns = pivot_weekly.columns.droplevel(0)
    # weekly_test=analysis.groupby([(analysis['total_factor'].abs()>2),'result_all']).agg(winning=('result_all','count'))
    # st.write(pivot_weekly)
    # st.write(weekly_test)


