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
factor_premier_league_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/premier_league_factor_result_2021_2022.csv','premier_league',2023)
factor_serie_a_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2022_2023.csv','serie_a',2023)
factor_serie_a_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/serie_a_factor_result_2021_2022.csv','serie_a',2022)
factor_bundesliga_2022_2023=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2022_2023.csv','bundesliga',2023)
factor_bundesliga_2021_2022=read_file('C:/Users/Darragh/Documents/Python/premier_league/bundesliga_factor_result_2021_2022.csv','bundesliga',2022)

# combine file above into one dataframe
combined_df=pd.concat([factor_la_liga_2022_2023,factor_la_liga_2021_2022,factor_premier_league_2021_2022,factor_premier_league_2022_2023,
factor_serie_a_2022_2023,factor_serie_a_2021_2022,factor_bundesliga_2022_2023,factor_bundesliga_2021_2022])

# st.write(combined_df)
# pivot table on dataframe
df_pivot=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?'],index=['index','season'],aggfunc=np.sum)
df_pivot_1=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?'],index=['index'],aggfunc=np.sum)
df_pivot_2=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?'],index=['index','league'],aggfunc=np.sum)
df_pivot_3=pd.pivot_table(combined_df,values=['total_turnover','total_season_cover','power_ranking_success?'],index=['index','league','season'],aggfunc=np.sum)

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
st.write(df_pivot)
st.write(df_pivot_1)
st.write(df_pivot_2)
st.write(df_pivot_3)