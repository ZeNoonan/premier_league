import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
st.title("📊 Premier League FBRef Weekly Movement Analysis")

BASE_PATH = r'C:/Users/Darragh/Documents/Python/premier_league/'
weeks = ['gw4', 'gw5', 'gw6', 'gw7']

# -----------------------------------------
# 🔹 Step 1 — Load and merge data functions
# -----------------------------------------
def read_fbref_file(file_path):
    """Reads the three sheets from a given Excel file."""
    sheets = ['Standard', 'Shots', 'Possession']
    usecols = {
        'Standard': ['Player', 'MP', 'npxG+xAG', 'Pos'],
        'Shots': ['Player', 'Sh'],
        'Possession': ['Player', 'Att Pen']
    }
    data = {}
    for sheet in sheets:
        data[sheet] = pd.read_excel(file_path, sheet_name=sheet, usecols=usecols[sheet])
    return data


def merge_sheets(sheets_dict):
    """
    Aggregates each sheet by Player before merging to avoid duplication
    for players appearing under multiple teams.
    """
    numeric_cols_by_sheet = {
        'Standard': ['MP', 'npxG+xAG'],
        'Shots': ['Sh'],
        'Possession': ['Att Pen']
    }

    aggregated_sheets = {}

    for sheet, df in sheets_dict.items():
        numeric_cols = numeric_cols_by_sheet[sheet]
        agg_map = {col: 'sum' for col in numeric_cols}
        if 'Pos' in df.columns:
            agg_map['Pos'] = 'first'

        grouped = df.groupby('Player', as_index=False).agg(agg_map)
        aggregated_sheets[sheet] = grouped

    # ✅ Merge AFTER grouping (now each sheet has unique Player)
    df = aggregated_sheets['Standard']
    df = df.merge(aggregated_sheets['Shots'], on='Player', how='outer')
    df = df.merge(aggregated_sheets['Possession'], on='Player', how='outer')

    # 🔍 Debug check for Eberechi
    # eze_check = df[df['Player'].str.contains('Eberechi', case=False, na=False)]
    # if not eze_check.empty:
    #     st.write("✅ Post-merge (grouped first) — 'Eberechi' aggregated correctly:")
    #     st.dataframe(eze_check)

    return df



# -----------------------------------------
# 🔹 Step 2 — Calculate movement (curr - prev)
# -----------------------------------------
def calc_movement(curr_df, prev_df):
    """Computes the week-on-week movement for numeric columns."""
    merged = pd.merge(curr_df, prev_df, on='Player', suffixes=('_curr', '_prev'), how='outer')

    for col in ['MP', 'Sh', 'npxG+xAG', 'Att Pen']:
        merged[col + '_wk'] = merged[col + '_curr'] - merged[col + '_prev']

    merged = merged.fillna(0)
    merged.loc[merged['MP_wk'] < 0, ['MP_wk', 'Sh_wk', 'npxG+xAG_wk', 'Att Pen_wk']] = 0

    return merged


# -----------------------------------------
# 🔹 Step 3 — Compute ranks for that movement
# -----------------------------------------
def compute_weekly_ranks(movement_df):
    """Computes ranks for the given week’s movement values."""
    # if position not available, fill as blank
    pos_col = 'Pos_curr' if 'Pos_curr' in movement_df.columns else 'Pos_prev'
    df = movement_df[['Player', pos_col, 'Sh_wk', 'npxG+xAG_wk', 'Att Pen_wk']].copy()
    df = df.rename(columns={pos_col: 'Pos'})

    df['Sh_rank'] = df['Sh_wk'].rank(ascending=False, method='dense')
    df['npxG+xAG_rank'] = df['npxG+xAG_wk'].rank(ascending=False, method='dense')
    df['Att Pen_rank'] = df['Att Pen_wk'].rank(ascending=False, method='dense')
    df['total_rank'] = df['Sh_rank'] + df['npxG+xAG_rank'] + df['Att Pen_rank']
    df['overall_rank'] = df['total_rank'].rank(ascending=True, method='dense')

    # Handle duplicates by taking best (lowest) overall rank per player
    ranks = df.groupby('Player', as_index=False)['overall_rank'].min().set_index('Player')
    moves = df.groupby('Player', as_index=False)[['Sh_wk', 'npxG+xAG_wk', 'Att Pen_wk']].sum().set_index('Player')

    return ranks, moves


# -----------------------------------------
# 🔹 Step 4 — Debug trace (for Eberechi)
# -----------------------------------------
st.subheader("🔍 Optional Debug View")
debug_toggle = st.checkbox("Run debug inspection for one week (shows intermediate data for 'Eberechi')", value=False)

if debug_toggle:
    prev_file = f"{BASE_PATH}fbref_stats_{weeks[0]}_done.xlsx"
    curr_file = f"{BASE_PATH}fbref_stats_{weeks[1]}_done.xlsx"
    week_label = weeks[1].upper()

    prev_sheets = read_fbref_file(prev_file)
    curr_sheets = read_fbref_file(curr_file)

    def show_player(df, label):
        subset = df[df['Player'].str.contains('Eberechi', case=False, na=False)]
        if not subset.empty:
            st.write(f"**{label} — showing rows for players containing 'Eberechi'**")
            st.dataframe(subset)
        else:
            st.info(f"No rows containing 'Eberechi' found in {label}")

    st.write(f"--- Debug trace for {week_label}: {prev_file.split('/')[-1]} → {curr_file.split('/')[-1]} ---")

    st.write("**📘 Previous File — Standard Sheet**")
    show_player(prev_sheets['Standard'], "Prev (Standard)")

    st.write("**📘 Current File — Standard Sheet**")
    show_player(curr_sheets['Standard'], "Curr (Standard)")

    prev_merged = merge_sheets(prev_sheets)
    curr_merged = merge_sheets(curr_sheets)

    st.write("**🔗 Merged Previous Week Data (after combining duplicates)**")
    show_player(prev_merged, "Prev (Merged)")

    st.write("**🔗 Merged Current Week Data (after combining duplicates)**")
    show_player(curr_merged, "Curr (Merged)")

    movement = calc_movement(curr_merged, prev_merged)
    st.write("**📊 Movement Calculations (curr - prev)**")
    show_player(movement[['Player', 'MP_wk', 'Sh_wk', 'npxG+xAG_wk', 'Att Pen_wk', 'Pos_curr']], "Movement")

    ranks, moves = compute_weekly_ranks(movement)

    st.write("**🏆 Weekly Rank for Eberechi**")
    if ranks.index.str.contains('Eberechi', case=False, na=False).any():
        st.write(ranks.loc[ranks.index.str.contains('Eberechi', case=False, na=False)])
    else:
        st.info("No player containing 'Eberechi' found in rank results")

    st.write("**📈 Movement Stats for Eberechi (Raw Weekly Values)**")
    st.dataframe(moves[moves.index.str.contains('Eberechi', case=False, na=False)])


# -----------------------------------------
# 🔹 Step 5 — Loop over all week pairs
# -----------------------------------------
week_ranks = {}
week_sh = {}
week_xag = {}
week_attpen = {}

for i in range(1, len(weeks)):
    prev_file = f"{BASE_PATH}fbref_stats_{weeks[i-1]}_done.xlsx"
    curr_file = f"{BASE_PATH}fbref_stats_{weeks[i]}_done.xlsx"
    gw_label = weeks[i].upper()

    prev_sheets = read_fbref_file(prev_file)
    curr_sheets = read_fbref_file(curr_file)

    prev_merged = merge_sheets(prev_sheets)
    curr_merged = merge_sheets(curr_sheets)
    movement = calc_movement(curr_merged, prev_merged)
    ranks, moves = compute_weekly_ranks(movement)

    week_ranks[gw_label] = ranks.rename(columns={'overall_rank': gw_label})
    week_sh[gw_label] = moves[['Sh_wk']].rename(columns={'Sh_wk': gw_label})
    week_xag[gw_label] = moves[['npxG+xAG_wk']].rename(columns={'npxG+xAG_wk': gw_label})
    week_attpen[gw_label] = moves[['Att Pen_wk']].rename(columns={'Att Pen_wk': gw_label})


# -----------------------------------------
# 🔹 Step 6 — Combine into final DataFrames
# -----------------------------------------
def combine_and_clean(dict_of_dfs, value_name):
    combined = pd.concat(dict_of_dfs, axis=1)
    combined.columns = [col[0] for col in combined.columns]  # flatten MultiIndex
    combined['Average'] = combined.mean(axis=1, skipna=True)
    combined = combined.dropna(how='all')  # remove rows with all None
    combined = combined.sort_values(by='Average', ascending=True)
    st.subheader(f"{value_name} — Weekly Movement with Average")
    st.dataframe(combined)
    return combined


st.divider()
st.header("🏁 Final Weekly Movement DataFrames")

rank_df = combine_and_clean(week_ranks, "Overall Rank")
sh_df = combine_and_clean(week_sh, "Shots (Sh)").sort_values(by='Average', ascending=False)
xag_df = combine_and_clean(week_xag, "npxG+xAG").sort_values(by='Average', ascending=False)
attpen_df = combine_and_clean(week_attpen, "Att Pen").sort_values(by='Average', ascending=False)

# -----------------------------------------
# 🔹 Step 7 — Player selector to compare across DataFrames
# -----------------------------------------
st.divider()
st.header("🎯 Player Comparison Dashboard")

# Combine all player names across DataFrames
all_players = sorted(
    set(rank_df.index)
    | set(sh_df.index)
    | set(xag_df.index)
    | set(attpen_df.index)
)

# Multi-select widget for players
selected_players = st.multiselect(
    "Select one or more players to compare:",
    options=all_players,
    default=["Eberechi Eze"],  # Default example — you can change this
    help="Pick players to view their rank and performance movements week by week."
)

# If any players selected, show their details
if selected_players:
    st.subheader("🏆 Overall Rank Movement")
    st.dataframe(rank_df.loc[selected_players])

    st.subheader("🔥 Shots (Sh) Weekly Movement")
    st.dataframe(sh_df.loc[selected_players])

    st.subheader("🎯 npxG + xAG Weekly Movement")
    st.dataframe(xag_df.loc[selected_players])

    st.subheader("⚡ Attacking Penetrations (Att Pen) Weekly Movement")
    st.dataframe(attpen_df.loc[selected_players])
else:
    st.info("👆 Use the selector above to choose one or more players for comparison.")
