import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import seaborn as sns

st.set_page_config(layout="wide")

@st.cache
def read_file(file,league='la_liga',season=2023):
    df=pd.read_csv(file)
    df['league']=league
    df['season']=season
    return df

factor_la_liga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_factor_result_2022_2023.csv','la_liga',2023)
factor_la_liga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_factor_result_2021_2022.csv','la_liga',2022)
factor_premier_league_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_factor_result_2021_2022.csv','premier_league',2022)
factor_premier_league_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_factor_result_2022_2023.csv','premier_league',2023)
factor_serie_a_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2022_2023.csv','serie_a',2023)
factor_serie_a_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2021_2022.csv','serie_a',2022)
factor_bundesliga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2022_2023.csv','bundesliga',2023)
factor_bundesliga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2021_2022.csv','bundesliga',2022)

betting_la_liga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_betting_result_2022_2023.csv','la_liga',2023)
betting_la_liga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/la_liga_betting_result_2021_2022.csv','la_liga',2022)
betting_premier_league_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_betting_result_2021_2022.csv','premier_league',2022)
betting_premier_league_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_betting_result_2022_2023.csv','premier_league',2023)
betting_serie_a_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_betting_result_2022_2023.csv','serie_a',2023)
betting_serie_a_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_betting_result_2021_2022.csv','serie_a',2022)
betting_bundesliga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_betting_result_2022_2023.csv','bundesliga',2023)
betting_bundesliga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_betting_result_2021_2022.csv','bundesliga',2022)


# combine file above into one dataframe
combined_df=pd.concat([factor_la_liga_2022_2023,factor_la_liga_2021_2022,factor_premier_league_2021_2022,factor_premier_league_2022_2023,
factor_serie_a_2022_2023,factor_serie_a_2021_2022,factor_bundesliga_2022_2023,factor_bundesliga_2021_2022])

combined_df_betting=pd.concat([betting_la_liga_2022_2023,betting_la_liga_2021_2022,betting_premier_league_2021_2022,betting_premier_league_2022_2023,
betting_serie_a_2022_2023,betting_serie_a_2021_2022,betting_bundesliga_2022_2023,betting_bundesliga_2021_2022])
# st.write('need to look at seeing what is in here',combined_df_betting)
# https://stackoverflow.com/questions/45803676/python-pandas-loc-filter-for-list-of-values
combined_df_betting=combined_df_betting.loc[combined_df_betting['category'].isin(['week_result'])]

st.write('2 need to look at seeing what is in here',combined_df_betting)
# pivot table on dataframe
df_pivot=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?','momentum_ranking_success?'],index=['index','season'],aggfunc=np.sum)
df_pivot_1=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?','momentum_ranking_success?'],index=['index'],aggfunc=np.sum)
df_pivot_2=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?','momentum_ranking_success?'],index=['index','league'],aggfunc=np.sum)
df_pivot_3=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?','momentum_ranking_success?'],index=['index','league','season'],aggfunc=np.sum)

df_pivot_betting=pd.pivot_table(combined_df_betting,values=['result'],index=['Week',],aggfunc=np.sum)
# st.write('pivot betting', df_pivot_betting)
df_pivot_1_betting=pd.pivot_table(combined_df_betting,values=['result'],index=['Week','league'],aggfunc=np.sum)
df_pivot_2_betting=pd.pivot_table(combined_df_betting,values=['result'],index=['Week','league','season'],aggfunc=np.sum).reset_index()
# st.write('pivot betting', df_pivot_2_betting)
df_pivot_2_betting['cum_result']=df_pivot_2_betting.groupby(['season','league'])['result'].cumsum()
df_pivot_2_betting['category']=df_pivot_2_betting['season'].astype(str)+'_'+df_pivot_2_betting['league'].astype(str) 
st.write('pivot betting after groupby', df_pivot_2_betting)
# 
graph_df_pivot_2_betting=df_pivot_2_betting.loc[:,['Week','category','cum_result']].rename(columns={'category':'season'})


df_pivot_3_betting=pd.pivot_table(combined_df_betting,values=['result'],index=['Week','season']).reset_index()
df_pivot_3_betting['cum_result']=df_pivot_3_betting.groupby(['season'])['result'].cumsum()
# st.write('pivot betting', df_pivot_3_betting)
# st.write(df_pivot_3_betting.groupby(['season'])['result'].cumsum())

def graph_pl_1(source,column):
    highlight = alt.selection(type='single', on='mouseover',fields=['season'], nearest=True)
    base = alt.Chart(source).encode(x='Week:O',y='cum_result',color='season:N')

    points = base.mark_circle().encode(opacity=alt.value(0)).add_selection(highlight).properties(width=600)

    lines = base.mark_line().encode(size=alt.condition(~highlight, alt.value(1), alt.value(3)))
    return st.altair_chart(points+lines ,use_container_width=True)

def graph_pl(decile_df_abs_home_1,column):
    # highlight = alt.selection(type='single', on='mouseover',fields=['season'], nearest=True)
    line_cover= alt.Chart(decile_df_abs_home_1).mark_line().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),alt.Y(column),color=alt.Color('season'))
    # points = line_cover.mark_circle().encode(opacity=alt.value(0)).add_selection(highlight).properties(width=600)
    # text_cover=line_cover.mark_text(baseline='middle',dx=0,dy=-15).encode(text=alt.Text(column),color=alt.value('black'))
    overlay = pd.DataFrame({column: [0]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=1).encode(y=column)
    # lines = line_cover.mark_line().encode(size=alt.condition(~highlight, alt.value(1), alt.value(3)))
    return st.altair_chart(line_cover+vline ,use_container_width=True)

def graph_pl_2(source):
    selection = alt.selection_multi(fields=['season'], bind='legend')
    return st.altair_chart( alt.Chart(source).mark_line().encode(
        alt.X('Week:O'),
        alt.Y('cum_result'),
        alt.Color('season:N'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection))

def graph_pl_3(source):
    highlight = alt.selection(type='single', on='mouseover', fields=['season'], nearest=True, bind='legend')
    selection = alt.selection_multi(fields=['season'], bind='legend', on='mouseover')
    base = alt.Chart(source).encode(x='Week:O',y='cum_result',color='season:N',tooltip=['season'])
    points = base.mark_circle().encode(opacity=alt.value(0.01)).add_selection(highlight).properties(width=1200)
    lines = base.mark_line().encode(
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),size=alt.condition(~highlight, alt.value(1), alt.value(3))).add_selection(selection)
    return st.altair_chart (points + lines, use_container_width=True)


graph_pl(df_pivot_3_betting,column='cum_result')
st.write('graph_df_pivot_2_betting', graph_df_pivot_2_betting)
graph_pl(graph_df_pivot_2_betting,column='cum_result')

# st.write('Graph of where line is highlighted')
# graph_pl_1(graph_df_pivot_2_betting,column='cum_result')

st.write('Graph of where legend highlights line')
graph_pl_2(graph_df_pivot_2_betting)
graph_pl_3(graph_df_pivot_2_betting)


df_pivot.loc[('% Winning',2022)] = (df_pivot.loc[('Winning_Bets',2022)] / (df_pivot.loc[('Winning_Bets',2022)]+df_pivot.loc[('Losing_Bets',2022)])  )
df_pivot.loc[('% Winning',2023)] = (df_pivot.loc[('Winning_Bets',2023)] / (df_pivot.loc[('Winning_Bets',2023)]+df_pivot.loc[('Losing_Bets',2023)])  )
df_pivot_2.loc[('% Winning','bundesliga')] = (df_pivot_2.loc[('Winning_Bets','bundesliga')] / (df_pivot_2.loc[('Winning_Bets','bundesliga')]+df_pivot_2.loc[('Losing_Bets','bundesliga')])  )
df_pivot_2.loc[('% Winning','la_liga')] = (df_pivot_2.loc[('Winning_Bets','la_liga')] / (df_pivot_2.loc[('Winning_Bets','la_liga')]+df_pivot_2.loc[('Losing_Bets','la_liga')])  )
df_pivot_2.loc[('% Winning','premier_league')] = (df_pivot_2.loc[('Winning_Bets','premier_league')] / (df_pivot_2.loc[('Winning_Bets','premier_league')]+df_pivot_2.loc[('Losing_Bets','premier_league')])  )
df_pivot_2.loc[('% Winning','serie_a')] = (df_pivot_2.loc[('Winning_Bets','serie_a')] / (df_pivot_2.loc[('Winning_Bets','serie_a')]+df_pivot_2.loc[('Losing_Bets','serie_a')])  )

df_pivot_3.loc[('% Winning','bundesliga',2022)] = (df_pivot_3.loc[('Winning_Bets','bundesliga',2022)] / (df_pivot_3.loc[('Winning_Bets','bundesliga',2022)]+df_pivot_3.loc[('Losing_Bets','bundesliga',2022)])  )
df_pivot_3.loc[('% Winning','bundesliga',2023)] = (df_pivot_3.loc[('Winning_Bets','bundesliga',2023)] / (df_pivot_3.loc[('Winning_Bets','bundesliga',2023)]+df_pivot_3.loc[('Losing_Bets','bundesliga',2023)])  )
df_pivot_3.loc[('% Winning','premier_league',2022)] = (df_pivot_3.loc[('Winning_Bets','premier_league',2022)] / (df_pivot_3.loc[('Winning_Bets','premier_league',2022)]+df_pivot_3.loc[('Losing_Bets','premier_league',2022)])  )
df_pivot_3.loc[('% Winning','premier_league',2023)] = (df_pivot_3.loc[('Winning_Bets','premier_league',2023)] / (df_pivot_3.loc[('Winning_Bets','premier_league',2023)]+df_pivot_3.loc[('Losing_Bets','premier_league',2023)])  )




df_pivot_1.loc['% Winning'] = (df_pivot_1.loc['Winning_Bets'] / (df_pivot_1.loc['Winning_Bets']+df_pivot_1.loc['Losing_Bets'])  )
st.write('Below is the result just split by season')
# st.write(df_pivot)
# filter dataframe by index
st.write(df_pivot.reset_index().set_index('index').loc[['% Winning','Winning_Bets','Losing_Bets','PL_Bets']].reset_index().set_index(['index','season'])\
    .style.format("{:.0f}", na_rep='-').format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :]))

st.write('Below is the result overall for the years')
st.write(df_pivot_1.reset_index().set_index('index').loc[['% Winning','Winning_Bets','Losing_Bets','PL_Bets']]\
    .style.format("{:.0f}", na_rep='-').format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :]))
st.write('Below is the result split by league for overall')
st.write(df_pivot_2.reset_index().set_index('index').loc[['% Winning','Winning_Bets','Losing_Bets','PL_Bets']].reset_index().set_index(['index','league'])\
    .style.format("{:.0f}", na_rep='-').format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :]))
st.write('Below is the result split by league and season')
st.write(df_pivot_3.reset_index().set_index('index').loc[['% Winning','Winning_Bets','Losing_Bets','PL_Bets']].reset_index().set_index(['index','league','season'])\
    .style.format("{:.0f}", na_rep='-').format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :]))