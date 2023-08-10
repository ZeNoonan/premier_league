import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")


# season_2023=pd.read_html('https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats#all_results2022-202391')[0]
# season_2022=pd.read_html('https://fbref.com/en/comps/9/2021-2022/2021-2022-Premier-League-Stats#all_results2021-202291')[0]
# season_2021=pd.read_html('https://fbref.com/en/comps/9/2020-2021/2020-2021-Premier-League-Stats#all_results2020-202191')[0]
# season_2020=pd.read_html('https://fbref.com/en/comps/9/2019-2020/2019-2020-Premier-League-Stats#all_results2019-202091')[0]
# season_2023.to_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2023.csv')
# season_2022.to_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2022.csv')
# season_2021.to_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2021.csv')
# season_2020.to_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2020.csv')

season_2023=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2023.csv')
season_2023['year']=2023
season_2022=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2022.csv')
season_2022['year']=2022
season_2021=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2021.csv')
season_2021['year']=2021
season_2020=pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_table_2020.csv')
season_2020['year']=2020
# st.write('df_1',season_2020)

combined_years=pd.concat([season_2023,season_2022,season_2021,season_2020],axis=0).reset_index(drop=True).sort_values(by='year',ascending=True)
cols=combined_years.columns
# st.write(cols)
weights_1 = np.array([0.125,0.25,0.5,1])
sum_weights_1 = np.sum(weights_1)
# combined_years['Weighted_ma_1'] = combined_years.groupby('team')['pts_per_game'].apply(lambda x: np.convolve(x, weights_1, 'valid') / sum_weights_1)
combined_years['Weighted_Pts'] = combined_years.groupby('Squad')['Pts'].apply(lambda x: x.rolling(window=len(weights_1), center=False)\
            .apply(lambda x: np.sum(weights_1*x) / sum_weights_1, raw=False))
combined_years['Weighted_xG'] = combined_years.groupby('Squad')['xGD'].apply(lambda x: x.rolling(window=len(weights_1), center=False)\
            .apply(lambda x: np.sum(weights_1*x) / sum_weights_1, raw=False))
combined_years['Weighted_GD'] = combined_years.groupby('Squad')['GD'].apply(lambda x: x.rolling(window=len(weights_1), center=False)\
            .apply(lambda x: np.sum(weights_1*x) / sum_weights_1, raw=False))
combined_years['Weighted_Pts_Rank']=combined_years['Weighted_Pts'].rank(method='dense', ascending=False)
combined_years['Weighted_xG_Rank']=combined_years['Weighted_xG'].rank(method='dense', ascending=False)
combined_years['Weighted_GD_Rank']=combined_years['Weighted_GD'].rank(method='dense', ascending=False)
avg_cols=['Weighted_Pts_Rank','Weighted_xG_Rank','Weighted_GD_Rank']
combined_years['Average_Rank']=combined_years[avg_cols].mean(axis=1)


combined_years=combined_years.sort_values(by=['year'],ascending=True)
combined_years['Weighted_Pts_Expo'] = combined_years.groupby('Squad')['Pts'].transform(lambda x: x.ewm(alpha=0.5).mean())
combined_years['Weighted_xG_Expo'] = combined_years.groupby('Squad')['xGD'].transform(lambda x: x.ewm(alpha=0.5).mean())
combined_years['Weighted_GD_Expo'] = combined_years.groupby('Squad')['GD'].transform(lambda x: x.ewm(alpha=0.5).mean())
combined_years['Weighted_Pts_Expo_Rank']=combined_years.groupby('year')['Weighted_Pts_Expo'].rank(method='dense', ascending=False)
combined_years['Weighted_xG_Expo_Rank']=combined_years.groupby('year')['Weighted_xG_Expo'].rank(method='dense', ascending=False)
combined_years['Weighted_GD_Expo_Rank']=combined_years.groupby('year')['Weighted_GD_Expo'].rank(method='dense', ascending=False)
avg_cols_expo=['Weighted_Pts_Expo_Rank','Weighted_xG_Expo_Rank','Weighted_GD_Expo_Rank']
combined_years['Average_Rank_Expo']=combined_years[avg_cols_expo].mean(axis=1)

st.write(   combined_years.loc[ (combined_years['year']==2023),['Squad','Average_Rank','Average_Rank_Expo'] ].sort_values(by='Average_Rank_Expo')   )

st.write('combined sorted on AB Weighted 4 Average', combined_years.set_index('Squad').sort_values(by=['year','Average_Rank'],ascending=[False,True]))
st.write('combined sorted on Expo same calc', combined_years.set_index('Squad').sort_values(by=['year','Average_Rank_Expo'],ascending=[False,True]))