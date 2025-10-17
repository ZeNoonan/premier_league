import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

standard = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw7_done.xlsx',sheet_name=['Standard'],usecols=['Player','MP','npxG+xAG','Pos'])
shots = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw7_done.xlsx',sheet_name=['Shots'],usecols=['Player','Sh'])
possession = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw7_done.xlsx',sheet_name=['Possession'],usecols=['Player','Att Pen'])

# standard = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw4_done.xlsx',sheet_name=['Standard'],usecols=['Player','MP','npxG+xAG','Pos'])
# shots = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw4_done.xlsx',sheet_name=['Shots'],usecols=['Player','Sh'])
# possession = pd.read_excel(f'C:/Users/Darragh/Documents/Python/premier_league/fbref_stats_gw4_done.xlsx',sheet_name=['Possession'],usecols=['Player','Att Pen'])

df = pd.merge(standard['Standard'], shots['Shots'], on='Player', how='outer')
df = pd.merge(df, possession['Possession'], on='Player', how='outer')


# rank the columns 'Sh', 'npxG+xAG', 'Att Pen' in descending order
df['Sh_rank'] = df['Sh'].rank(ascending=False, method='dense')
df['npxG+xAG_rank'] = df['npxG+xAG'].rank(ascending=False, method='dense')
df['Att Pen_rank'] = df['Att Pen'].rank(ascending=False, method='dense')    

# total the ranks of 'Sh_rank', 'npxG+xAG_rank', 'Att Pen_rank' into a new column 'total_rank'
df['total_rank'] = df['Sh_rank'] + df['npxG+xAG_rank'] + df['Att Pen_rank'] 
# rank the 'total_rank' column in ascending order
df['overall_rank'] = df['total_rank'].rank(ascending=True, method='dense')

# hide rows where 'total_rank' is NaN
df = df[df['total_rank'].notna()]

st.title('Premier League Shot Stats')
# sort by overall_rank
df = df.sort_values(by='overall_rank',ascending=True)
st.write(df.set_index('Player'))

names_to_find = ['bowen', 'bruno Fernandes', 'Ismaila Sarr','Bryan Mbeumo','Evanilson','Semenyo','João Pedro','Jack Grealish']
filter_condition = df['Player'].str.contains('|'.join(names_to_find), case=False, na=False)
st.write(df[filter_condition])
# filter the df so that column 'Pos' contains only 'DF'

st.write('Defenders', df[df['Pos'].str.contains( 'DF')] )


df['Sh_per_MP'] = df['Sh'] / df['MP']
df['npxG+xAG_per_MP'] = df['npxG+xAG'] / df['MP']
df['Att Pen_per_MP'] = df['Att Pen'] / df['MP']

df['Sh_per_MP_rank'] = df['Sh_per_MP'].rank(ascending=False, method='dense')
df['npxG+xAG_per_MP_rank'] = df['npxG+xAG_per_MP'].rank(ascending=False, method='dense')
df['Att Pen_per_MP_rank'] = df['Att Pen_per_MP'].rank(ascending=False, method='dense')
df['total_rank_per_MP'] = df['Sh_per_MP_rank'] + df['npxG+xAG_per_MP_rank'] + df['Att Pen_per_MP_rank']

# filter the df so that 'MP' column is greater than or equal to 3
df = df[df['MP'] >= 3]

# display the dataframe focusing on the per MP ranks
st.write('Per Minute Played Ranks', df[['Player','MP','Sh_per_MP','npxG+xAG_per_MP','Att Pen_per_MP','total_rank_per_MP']].sort_values(by='total_rank_per_MP',ascending=True).set_index('Player'))
st.write(df[filter_condition].set_index('Player')[['MP','Sh_per_MP','npxG+xAG_per_MP','Att Pen_per_MP','total_rank_per_MP']])

# show the above dataframe but only for defenders
st.write('Defenders Per Minute Played Ranks', df[df['Pos'].str.contains( 'DF')][['Player','MP','Sh_per_MP','npxG+xAG_per_MP','Att Pen_per_MP','total_rank_per_MP']].sort_values(by='total_rank_per_MP',ascending=True).set_index('Player'))