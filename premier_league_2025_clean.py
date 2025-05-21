import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import seaborn as sns

st.set_page_config(layout="wide")
dfa=pd.read_html('https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures')
dfa[0].to_csv('C:/Users/Darragh/Documents/Python/soccer/premier_league_scores_2024_2025.csv')
points_table=pd.read_html('https://fbref.com/en/comps/9/Premier-League-Stats')[0]
points_table.to_csv('C:/Users/Darragh/Documents/Python/soccer/premier_league_table_2024_2025.csv')

df=pd.read_csv('C:/Users/Darragh/Documents/Python/soccer/premier_league_scores_2024_2025.csv',parse_dates=['Date'])

odds=pd.read_excel('C:/Users/Darragh/Documents/Python/soccer/premier_league_odds_2024_2025.xlsx')
odds_csv=odds.to_csv('C:/Users/Darragh/Documents/Python/soccer/premier_league_odds_2024_2025.csv')
odds=odds.rename(columns={'Home Team':'Home','Away Team':'Away'})
# st.write(odds.head())



merged_df = pd.merge(df,odds,on=['Date','Home','Away'],how='outer',indicator=True)\
.drop(['Day_y','Wk_x','Notes','Match Report','Referee','Venue','Attendance','Unnamed: 0','Time'],axis=1)\
    .rename(columns={'xG_x':'xG','xG.1_x':'xG.1','Score_x':'Score','Day_x':'Day','Date_x':'Date','Home':'Home Team','Away':'Away Team','Wk_y':'Week'})\
    .drop(['Day'],axis=1)

with st.expander('Check Merge'):
    # check Merge
    not_both_in_indicator = merged_df['_merge'] != 'both'
    # Output the rows where '_merge' is not 'both'
    result = merged_df[not_both_in_indicator]
    st.write(result.sort_values(by=['Date']))

merged_df['Home Points'] = [str(x)[0] for x in merged_df['Score']]
merged_df['Away Points'] = [str(x)[2] for x in merged_df['Score']]
merged_df['home_spread']=merged_df['Spread']
merged_df['away_spread']=-merged_df['Spread']
merged_df['Home Points']=merged_df['Home Points'].replace({'n':np.nan})
merged_df['Away Points']=merged_df['Away Points'].replace({'n':np.nan})
merged_df['Home Points']=pd.to_numeric(merged_df['Home Points'])
merged_df['Away Points']=pd.to_numeric(merged_df['Away Points'])
merged_df['Home_Total_Points'] = (merged_df['Closing_Total']-merged_df['Spread'] )/2
merged_df['Away_Total_Points'] = (merged_df['Closing_Total']+merged_df['Spread'] )/2

def spread_workings(data):
    data['home_win']=data['Home Points'] - data['Away Points']
    data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
    
    data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
    
    data['first_spread']=(np.where(((data['Home Points'] + (data['Spread']+0.25)) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']+.25) < data['Away Points']),-1,0)))*.5
    data['second_spread']=(np.where(((data['Home Points'] + (data['Spread']-0.25)) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']-0.25) < data['Away Points']),-1,0)))*.5
    data['home_cover']=data['first_spread']+data['second_spread']

    # data['home_cover']=data['home_cover'].astype(int)
    data['home_cover']=data['home_cover'].astype(float)
    data['away_cover'] = -data['home_cover']
    return data

def spread_workings_total(data):
    data['home_cover_total']=(np.where(((data['Home Points'] + data['Away Points']) > data['Closing_Total']), 1,
    np.where(((data['Home Points']+ data['Away Points']) < data['Closing_Total']),-1,0)))
    data['home_cover_total']=data['home_cover_total'].astype(int)
    data['away_cover_total'] = data['home_cover_total'] # THIS IS RIGHT I THINK AS HOME AND AWAY ARE SAME
    return data

def spread_workings_total_team(data):
    data['home_cover_total_team']=(np.where(((data['Home Points'] ) > data['Home_Total_Points']), 1,
    np.where(((data['Home Points']) < data['Home_Total_Points']),-1,0)))
    data['home_cover_total_team']=data['home_cover_total_team'].astype(int)
    data['away_cover_total_team']=(np.where(((data['Away Points'] ) > data['Away_Total_Points']), 1,
    np.where(((data['Away Points']) < data['Away_Total_Points']),-1,0)))
    data['away_cover_total_team']=data['away_cover_total_team'].astype(int)

    return data

spread=spread_workings(merged_df)
spread=spread_workings_total(spread)
spread=spread_workings_total_team(spread)
# st.write('check this for totals workings',spread)

team_names_id=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_team_names_id_2024_2025.csv')
team_names_id=team_names_id.rename(columns={'team':'Home Team'})
# check against other py file
odds_data=pd.merge(spread,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'}).sort_values(by='Date',ascending=False)
team_names_id=team_names_id.rename(columns={'Home Team':'Away Team'})
odds_data=pd.merge(odds_data,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'}).sort_values(by='Date',ascending=False)
# st.write('odds data are IDs in hereer??', odds_data)
odds_data=odds_data.sort_values(by=['Date','Home Team'],ascending=[True,True]).reset_index(drop=True).drop(['Unnamed: 0_y','Unnamed: 0_x'],axis=1)
# st.write('odds data', odds_data)
spread=odds_data.reset_index().rename(columns={'index':'unique_match_id'})
# st.write('odds data', odds_data.head())






def season_cover_workings_total(data,home,away,name,week_start):
    season_cover_df=data[data['Week']>week_start].copy()
    home_cover_df = (season_cover_df.loc[:,['Week','Date','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    away_cover_df = (season_cover_df.loc[:,['Week','Date','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=[True,True,True])

def season_cover_2_total(season_cover_df,column_name):    
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    return season_cover_df

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

def season_cover_2_updated(spead_1):  
    x=spread_1.groupby (['ID'])['cover'].apply(lambda x: x.cumsum().shift()).reset_index().rename(columns={'level_1':'unique_id'})
    # st.write('in function and applying shift',x)
    y=spread_1.drop('cover',axis=1).reset_index().rename(columns={'index':'unique_id'})
    # st.write('dropping cover in function',y.reset_index().rename(columns={'index':'unique_id'}))
    # st.write('dropping cover in function',y)
    season_cover_df=pd.merge(x,y,how='outer',on=['unique_id','ID'])
    # st.write('after merge', season_cover_df)
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop(['index','unique_id'],axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

def season_cover_2_updated_again(spread_1,col_selection='cover_total_team'):  
    x=spread_1.groupby (['ID'])[col_selection].apply(lambda x: x.cumsum().shift())\
        .reset_index().rename(columns={'level_1':'unique_id'})
    y=spread_1.drop(col_selection,axis=1).reset_index().rename(columns={'index':'unique_id'})
    season_cover_df=pd.merge(x,y,how='outer',on=['unique_id','ID'])
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True)\
        .drop(['index','unique_id'],axis=1)
    return season_cover_df

def season_cover_2_updated_graph(spread_1):  
    x=spread_1.groupby (['ID'])['cover'].apply(lambda x: x.cumsum()).reset_index().rename(columns={'level_1':'unique_id'})
    y=spread_1.drop('cover',axis=1).reset_index().rename(columns={'index':'unique_id'})
    season_cover_df=pd.merge(x,y,how='outer',on=['unique_id','ID'])
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop(['index','unique_id'],axis=1)
    return season_cover_df

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

# st.write('this is the base data....check', matrix_df)
spread_1 = season_cover_workings(odds_data,'home_cover','away_cover','cover',0)
spread_1_total_team = season_cover_workings(spread,'home_cover_total_team','away_cover_total_team','cover_total_team',0)
spread_1_total = season_cover_workings_total(spread,'home_cover_total','away_cover_total','cover_total',0)
# st.write('testing',spread_1_total_team.groupby (['ID'])['cover_total_team'].apply(lambda x: x.cumsum().shift())\
#         .reset_index().rename(columns={'level_1':'unique_id'}))

# st.write('spread total ',spread_1_total_team, 'cols',spread_1_total_team.columns, 'check',spread_1_total_team['cover_total_team'])
spread_2_total_team=season_cover_2_updated_again(spread_1_total_team,col_selection='cover_total_team')

spread_2_total=season_cover_2_updated_again(spread_1_total,col_selection='cover_total')
spread_2=season_cover_2_updated(spread_1)
# st.write('spread 2 should this not be adding up to 0.5 increments for some of them', spread_2)
spread_3=season_cover_3(spread_2,'cover_sign','cover')
spread_3_total_team=season_cover_3(spread_2_total_team,'cover_sign_total_team','cover_total_team')
spread_3_total=season_cover_3(spread_2_total,'cover_sign_total','cover_total')

spread_2_graph=season_cover_2_updated_graph(spread_1)
spread_3_graph=season_cover_3(spread_2_graph,'cover_sign','cover')

with st.expander('Season to Date Cover Graph'):
    st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    st.write('blanks in graph are where the team got a bye week')
    # st.write('spread 3', spread_3,'team id', team_names_id) 
    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=spread.drop(['away_cover'],axis=1)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    # st.write('problem below here this is "updated_df" ',updated_df)
    # st.write('this is the "stdc_home" ',stdc_home)
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
    # st.write(spread_3)
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc 1',df_stdc_1)
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    df_stdc_1_graph=pd.merge(spread_3_graph,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1_graph=df_stdc_1_graph.loc[:,['Week','ID','Team','cover']].copy()
    df_stdc_1_graph['average']=df_stdc_1_graph.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc graph',df_stdc_1_graph)
    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    # st.altair_chart(chart_cover + text_cover,use_container_width=True)

    chart_cover= alt.Chart(df_stdc_1_graph).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

# st.write('spread 3', spread_3)
with st.expander('Season to Date Team Total Cover Graph'):
    # st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    # st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    # st.write('blanks in graph are where the team got a bye week')
    # st.write('season', spread_3_total_team)
    stdc_home=spread_3_total_team.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign_total_team']=-stdc_home['cover_sign_total_team']
    stdc_away=spread_3_total_team.rename(columns={'ID':'Away ID'})
    # st.write('update', updated_df)
    # updated_df=spread.drop(['away_cover_total_team'],axis=1)
    # updated_df=updated_df.rename(columns={'home_cover_total_team':'home_cover_result'})
    # updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign_total_team':'home_cover_sign'})
    # updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign_total_team':'away_cover_sign'})
    stdc_df=pd.merge(spread_3_total_team,team_names_id,on='ID').rename(columns={'Away Team':'Team','cover_total_team':'cover'})
    team_names_id_update=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    df_stdc_1=pd.merge(spread_3_total_team,team_names_id_update,on='ID').rename(columns={'Away Team':'Team','cover_total_team':'cover'})
    # st.write('stdc',df_stdc_1)
    df_stdc_1=df_stdc_1.loc[:,['Week','ID','Team','cover']].copy()
    # st.write(spread_3)
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc 1',df_stdc_1)
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    df_stdc_1_graph=pd.merge(spread_3_graph,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1_graph=df_stdc_1_graph.loc[:,['Week','ID','Team','cover']].copy()
    df_stdc_1_graph['average']=df_stdc_1_graph.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc graph',df_stdc_1_graph)
    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

    # chart_cover= alt.Chart(df_stdc_1_graph).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # # https://vega.github.io/vega/docs/schemes/
    # text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    # st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Season to Date Total Cover Graph'):
    # st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    # st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    # st.write('blanks in graph are where the team got a bye week')
    # st.write('season', spread_3_total_team)
    # stdc_home=spread_3_total.rename(columns={'ID':'Home ID'})
    st.write(stdc_home)
    # stdc_home['cover_sign_total_team']=-stdc_home['cover_sign_total_team']
    # stdc_away=spread_3_total_team.rename(columns={'ID':'Away ID'})
    # st.write('update', updated_df)
    # updated_df=spread.drop(['away_cover_total_team'],axis=1)
    # updated_df=updated_df.rename(columns={'home_cover_total_team':'home_cover_result'})
    # updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign_total_team':'home_cover_sign'})
    # updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign_total_team':'away_cover_sign'})
    stdc_df=pd.merge(spread_3_total,team_names_id,on='ID').rename(columns={'Away Team':'Team','cover_total':'cover'})
    st.write(stdc_df)
    # team_names_id_update=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    # df_stdc_1=pd.merge(spread_3_total,team_names_id_update,on='ID').rename(columns={'Away Team':'Team','cover_total_team':'cover'})
    # st.write('stdc',df_stdc_1)
    df_stdc_1=df_stdc_1.loc[:,['Week','ID','Team','cover']].copy()
    # st.write(spread_3)
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc 1',df_stdc_1)
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    df_stdc_1_graph=pd.merge(spread_3_graph,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1_graph=df_stdc_1_graph.loc[:,['Week','ID','Team','cover']].copy()
    df_stdc_1_graph['average']=df_stdc_1_graph.groupby('Team')['cover'].transform(np.mean)
    # st.write('stdc graph',df_stdc_1_graph)
    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)


with st.expander('Handicap Analysis'):

    points_table=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2024_2025.csv')\
        .loc[:,['Squad','MP','Pts']].rename(columns={'Squad':'team'})


    handicap_df=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league_odds_2024_2025.xlsx',sheet_name='handicap')
    # st.write(points_table)
    # st.write(handicap_df)
    merged_df_handicap = pd.merge(handicap_df,points_table,on='team',how='outer',indicator=True)
    merged_df_handicap['total_points']=merged_df_handicap['Pts']+merged_df_handicap['handicap']
    merged_df_handicap=merged_df_handicap.drop(['ID','_merge','MP'],axis=1)
    # merged_df_handicap['ranking']=merged_df_handicap['total_points'].rank(ascending=False)
    # sort dataframe by total points
    merged_df_handicap=merged_df_handicap.sort_values(by='total_points',ascending=False).reset_index(drop=True)
    # st.write(merged_df_handicap)
    def color_recommend(value):
        # if value == 'Fulham':
        #     color = 'red'
        if value in ['Brighton', 'Brentford','Fulham']:
            color = 'lightgreen'
        else:
            return
        return f'background-color: {color}'
    
    st.write(merged_df_handicap.style.applymap(color_recommend, subset=['team']))
