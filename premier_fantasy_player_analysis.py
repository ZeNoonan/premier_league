import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import datetime as dt

st.set_page_config(layout="wide")


name = 'closing'
series = pd.Series([2, 2, 6, 2, 8, 6, 1, 6,2, 2, 6, 2, 8, 6, 1, 6], name=name).to_frame()
# period = 4
period = 3
# alpha = 2/(1+period)
alpha = 0.15
# alpha = 0.93

# weights = np.array([0.125,0.125, 0.25,0.25,0.5,0.5,1,1]) # the order mattered!! took me a while to figure this out
weights = np.array([0.007813,0.015625,0.03125,0.0625,0.125,0.25,0.5,1])
weights_1 = np.array([0.125,0.25,0.5,1])
# st.write('weights', len(weights)
weights_2 = np.array([0.321,0.377,0.444,0.522,0.614,0.723,0.85,1])
sum_weights = np.sum(weights)
sum_weights_1 = np.sum(weights_1)
sum_weights_2 = np.sum(weights_2)

series['Weighted_ma'] = (series['closing'].fillna(0).rolling(window=len(weights), center=False)\
    .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)) # raw=False

series['Weighted_ma_1'] = (series['closing'].fillna(0).rolling(window=len(weights_1), center=False)\
    .apply(lambda x: np.sum(weights_1*x) / sum_weights_1, raw=False)) # raw=False

series['Weighted_ma_2'] = (series['closing'].fillna(0).rolling(window=len(weights_2), center=False)\
    .apply(lambda x: np.sum(weights_2*x) / sum_weights_2, raw=False)) # raw=False


series[name+'_ewma'] = np.nan
series.loc[0, name+'_ewma'] = series[name].iloc[0]

# st.write('series',  series)

series[name+'_ewma_adjust'] = np.nan
series.loc[0, name+'_ewma_adjust'] = series[name].iloc[0]
series['mean']=series['closing'].expanding().mean()
# https://stackoverflow.com/questions/37924377/does-pandas-calculate-ewm-wrong

for i in range(1, len(series)):
    # st.write('i',i) # interesting use of i, could be useful with opening and closing balances, reminds me of the knapsack problem
    series.loc[i, name+'_ewma'] = (1-alpha) * series.loc[i-1, name+'_ewma'] + alpha * series.loc[i, name]
    # st.write('i',i,'series.loc[i, name+_ewma]',series.loc[i, name+'_ewma'])
    series.loc[i,'i']=i
    series.loc[i, "series.loc[i-1, name+'_ewma']"]= series.loc[i-1, name+'_ewma']
    series.loc[i,'(1-alpha)']=(1-alpha)
    series.loc[i,'multiply_previous_columns']=series.loc[i,'(1-alpha)'] * series.loc[i, "series.loc[i-1, name+'_ewma']"]

    series.loc[i,'name']=series.loc[i, name]
    series.loc[i,'alpha']=alpha
    series.loc[i,'multiply_prev_cols']=series.loc[i,'name'] * series.loc[i,'alpha']

    series.loc[i,'add_multiplied_cols'] = series.loc[i,'multiply_previous_columns'] + series.loc[i,'multiply_prev_cols']
    series.loc[i,'check_diff'] = series.loc[i,'add_multiplied_cols'] - series.loc[i, name+'_ewma']

    ajusted_weights = np.array([(1-alpha)**(i-t) for t in range(i+1)])
    
    series.loc[i, name+'_ewma_adjust'] = np.sum(series.iloc[0:i+1][name].values * ajusted_weights)\
         / ajusted_weights.sum()


# series['1-alpha']=1-alpha
# series['test']=series['closing'].ewm(alpha=0.4,adjust=True).mean()
# series['test_1']=series['closing'].ewm(alpha=0.4,adjust=False).mean()

st.write(series)
st.write("diff adjusted=False -> ", np.sum(series[name+'_ewma'] - series[name].ewm(span=period, adjust=False).mean()))
st.write("diff adjusted=True -> ",int( np.sum(series[name+'_ewma_adjust'] - series[name]\
    .ewm(span=period, adjust=True).mean())  )  )


data = pd.read_csv('C:/Users/Darragh/Documents/Python/premier_league/premier_league_data_fpl.csv')

def player_data_generation(data,name='bruyne'):
    bruyne=data[data['full_name'].str.contains(name)]\
        .loc[:,['full_name','Game_1','week','year','minutes','Clean_Pts','last_38_ppg','last_19_ppg']]\
            .sort_values(by=['year','week'],ascending=[True,True]).copy()
    # bruyne['date']=bruyne['year'].astype(str)+'-'+bruyne['week'].astype(str)
    st.write('need to combine a year plus week into date format')
    # bruyne['date']=
    bruyne=bruyne.reset_index(drop=True).reset_index().rename(columns={'index':'date'})

    # bruyne['formatted_date'] = bruyne['year'] * 1000 + bruyne['week'] * 10 + 0
    # bruyne['date'] = pd.to_datetime(bruyne['formatted_date'], format='%Y%W%w')
    # https://stackoverflow.com/questions/55222420/converting-year-and-week-of-year-columns-to-date-in-pandas
    bruyne['mean']=bruyne['Clean_Pts'].expanding().mean()
    bruyne['expo_mean_0.15']=bruyne['Clean_Pts'].ewm(alpha=0.15,adjust=False).mean()
    bruyne['expo_0.25']=bruyne['Clean_Pts'].ewm(alpha=0.25,adjust=False).mean()
    bruyne['expo_0.06']=bruyne['Clean_Pts'].ewm(alpha=0.06,adjust=False).mean()
    bruyne['19_game_rolling_avg']=bruyne['Clean_Pts'].rolling(window=19,min_periods=1).mean()
    bruyne['38_game_rolling_avg']=bruyne['Clean_Pts'].rolling(window=38,min_periods=1).mean()
    bruyne['76_game_rolling_avg']=bruyne['Clean_Pts'].rolling(window=76,min_periods=1).mean()
    bruyne['Weighted_ma_AB_4'] = (bruyne['Clean_Pts'].fillna(0).rolling(window=len(weights_1), center=False)\
        .apply(lambda x: np.sum(weights_1*x) / sum_weights_1, raw=False)) # raw=False

    bruyne['Weighted_ma_0.85'] = (bruyne['Clean_Pts'].fillna(0).rolling(window=len(weights_2), center=False)\
        .apply(lambda x: np.sum(weights_2*x) / sum_weights_2, raw=False)) # raw=False
    return bruyne

bruyne = player_data_generation(data,name='bruyne')
son = player_data_generation(data,name='heung')
almiron = player_data_generation(data,name='almir')
st.write('bruyne',bruyne)
st.write('son',son)
st.write('almiron',almiron)
st.write('interesting to see what Son looks like still ranked high')

def generate_loc_df(bruyne):
    graph_data=bruyne[['date','mean','expo_0.06','expo_mean_0.15','expo_0.25','Weighted_ma_AB_4','Weighted_ma_0.85']].copy()
    # graph_data_1=bruyne[['date','mean','expo_0.06','19_game_rolling_avg','38_game_rolling_avg','76_game_rolling_avg']].copy()
    graph_data_1=bruyne[['date','mean','expo_0.06','19_game_rolling_avg']].copy()
    # graph_data_1=bruyne[['date','mean','expo_0.06','76_game_rolling_avg']].copy()
    return graph_data,graph_data_1
graph_data,graph_data_1=generate_loc_df(bruyne)
graph_data_son,graph_data_1_son=generate_loc_df(son)
graph_data_almiron,graph_data_1_almiron=generate_loc_df(almiron)

# st.write('graph_data',graph_data.head())
# https://stackoverflow.com/questions/68961796/how-do-i-melt-a-pandas-dataframe

graph_data=graph_data.melt(id_vars=['date'],var_name='symbol',value_name='price')
graph_data_1=graph_data_1.melt(id_vars=['date'],var_name='symbol',value_name='price')

graph_data_son=graph_data_son.melt(id_vars=['date'],var_name='symbol',value_name='price')
graph_data_1_son=graph_data_1_son.melt(id_vars=['date'],var_name='symbol',value_name='price')

graph_data_almiron=graph_data_almiron.melt(id_vars=['date'],var_name='symbol',value_name='price')
graph_data_1_almiron=graph_data_1_almiron.melt(id_vars=['date'],var_name='symbol',value_name='price')


# st.write('graph_data',graph_data)

def draw_graph(graph_data, start=4, end=7):
    graph=alt.Chart(graph_data).mark_line().encode(
        x='date',
        y=alt.Y('price',scale=alt.Scale(domain=(start, end),clamp=True)),
        # y='price',
        color='symbol',
        strokeDash='symbol')
    return graph

graph=draw_graph(graph_data)
st.altair_chart(graph, use_container_width=True)

graph=draw_graph(graph_data_1)
st.altair_chart(graph, use_container_width=True)

st.write('Son graph below')
graph=draw_graph(graph_data_1_son)
st.altair_chart(graph, use_container_width=True)

st.write('Almirons graph below')
graph=draw_graph(graph_data_1_almiron, start=1, end=6)
st.altair_chart(graph, use_container_width=True)

st.write('Almirons graph below')
graph=draw_graph(graph_data_almiron, start=1, end=10)
st.altair_chart(graph, use_container_width=True)