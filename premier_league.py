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
# st.write('merged', merged_df)

team_names_id=pd.read_excel('C:/Users/Darragh/Documents/Python/premier_league/premier_league.xlsx',sheet_name='Sheet2')

st.write(team_names_id)
st.write('merged', merged_df)

# st.write(odds)
# st.write(df)



