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

with st.beta_expander('df'):
    # dfa=pd.read_html('https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures')
    # dfa[0].to_pickle('C:/Users/Darragh/Documents/Python/premier_league/scores.pkl')
    # dfa[0].to_csv('C:/Users/Darragh/Documents/Python/premier_league/scores.csv')
    df=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/scores.csv')
    # st.write(df)

    df=df.dropna(subset=['Wk'])
    # st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    odds = pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx')
    prior_data=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/prior_year.xlsx')
    def concat_current_prior(x,y):
        current_plus_prior = pd.concat([x,y],axis=0,ignore_index=True)
        return current_plus_prior

    odds=concat_current_prior(odds,prior_data)
    # st.write(odds)
    merged_df = pd.merge(df,odds,on=['Wk','Home','Away'],how='outer').drop(['xG_y','Score_y','xG.1_y','Day_y','Date_y'],axis=1)\
        .rename(columns={'xG_x':'xG','xG.1_x':'xG.1','Score_x':'Score','Day_x':'Day','Date_x':'Date','Home':'Home Team','Away':'Away Team','Wk':'Week'})
    # st.write('merged df',merged_df)
    # https://stackoverflow.com/questions/35552874/get-first-letter-of-a-string-from-column
    merged_df['Home Points'] = [str(x)[0] for x in merged_df['Score']]
    merged_df['Away Points'] = [str(x)[2] for x in merged_df['Score']]
    merged_df['home_spread']=-merged_df['Spread']
    merged_df['away_spread']=merged_df['Spread']
    merged_df=merged_df[merged_df['Notes']!='Match Postponed']
    merged_df['Home Points']=merged_df['Home Points'].replace({'n':np.NaN})
    merged_df['Away Points']=merged_df['Away Points'].replace({'n':np.NaN})
    merged_df['Home Points']=pd.to_numeric(merged_df['Home Points'])
    merged_df['Away Points']=pd.to_numeric(merged_df['Away Points'])
    # st.write(merged_df.dtypes)
    data=merged_df
    # st.write(data)

    def spread_workings(data):
        data['home_win']=data['Home Points'] - data['Away Points']
        data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
        data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
        np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
        data['home_cover']=data['home_cover'].astype(int)
        data['away_cover'] = -data['home_cover']
        return data

    spread=spread_workings(data)
    # st.write(spread)

    team_names_id=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx', sheet_name='Sheet2')
    # st.write(team_names_id)
    team_names_id=team_names_id.rename(columns={'team':'Home Team'})
    odds_data=pd.merge(spread,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'}).sort_values(by='Date',ascending=False)
    team_names_id=team_names_id.rename(columns={'Home Team':'Away Team'})
    odds_data=pd.merge(odds_data,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'}).sort_values(by='Date',ascending=False)
    st.write(odds_data)

    matrix_df=odds_data.reset_index().rename(columns={'index':'unique_match_id'})
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
    st.write(full_stack)

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
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

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

spread_1 = season_cover_workings(matrix_df,'home_cover','away_cover','cover',0)
spread_2=season_cover_2(spread_1,'cover')
spread_3=season_cover_3(spread_2,'cover_sign','cover')

with st.beta_expander('Season to Date Cover Graph'):
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
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.beta_expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    team_names_id=team_names_id.rename(columns={'Away Team':'Team'})
    id_names=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    pivot_df=pd.merge(power_week,id_names, on='ID')
    # st.write('after merge', pivot_df)
    pivot_df=pivot_df.loc[:,['Team','final_power','week']].copy()
    # st.write('graphing?',pivot_df)
    power_pivot=pd.pivot_table(pivot_df,index='Team', columns='week')
    pivot_df_test = pivot_df.copy()
    pivot_df_test=pivot_df_test[pivot_df_test['week']<19]
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
    text=chart_power.mark_text().encode(text=alt.Text('final_power:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)

