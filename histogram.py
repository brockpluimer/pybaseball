import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any
from load_data import load_and_filter_data
from colors import load_team_colors, get_team_color

def plot_league_wide_stat(df, stat, year_range, stat_min_max, hover_data, data_type):
    if stat not in df.columns:
        st.error(f"The stat '{stat}' is not available in the dataset.")
        return

    valid_years = df[df[stat].notnull()]['year']
    if valid_years.empty:
        st.error(f"No valid data available for the stat '{stat}'.")
        return

    first_year_available = valid_years.min()
    last_year_available = valid_years.max()

    adjusted_start_year = max(year_range[0], first_year_available)
    adjusted_end_year = min(year_range[1], last_year_available)

    if adjusted_start_year > adjusted_end_year:
        st.warning(f"No data available for '{stat}' in the selected year range. Adjusting to available years.")
        adjusted_start_year = first_year_available
        adjusted_end_year = last_year_available

    df_filtered = df[(df['year'] >= adjusted_start_year) & (df['year'] <= adjusted_end_year)]

    if stat_min_max == 'min':
        idx = df_filtered.groupby('year')[stat].idxmin()
    else:
        idx = df_filtered.groupby('year')[stat].idxmax()

    grouped_stat = df_filtered.loc[idx].reset_index(drop=True)

    team_colors = load_team_colors()

    def format_value(value, key):
        if pd.isna(value):
            return "N/A"
        if key in ['AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'wOBA']:
            return f"{value:.3f}"
        elif key in ['ERA', 'FIP', 'xFIP', 'WHIP']:
            return f"{value:.2f}"
        elif key in ['HR', 'R', 'RBI', 'SB', 'BB', '1B', '2B', '3B', 'SO', 'H', 'G', 'GS', 'W', 'L', 'SV']:
            return f"{int(value)}"
        else:
            return f"{value:.1f}"

    hover_text = grouped_stat.apply(lambda row: (
        f"Name: {row['Name']}<br>"
        f"Year: {row['year']}<br>"
        f"Team: {row['Team']}<br>"
        f"{stat}: {format_value(row[stat], stat)}<br>" +
        ''.join(f"{key}: {format_value(row[key], key)}<br>" for key in hover_data if key in row.index and key != stat)
    ), axis=1)

    fig = go.Figure()

    for team in sorted(grouped_stat['Team'].unique()):
        team_data = grouped_stat[grouped_stat['Team'] == team]
        fig.add_trace(go.Bar(
            x=team_data['year'],
            y=team_data[stat],
            name=team,
            marker_color=get_team_color(team, team_colors),
            hoverinfo='text',
            hovertext=hover_text[team_data.index],
            text=team_data[stat].apply(lambda x: format_value(x, stat)),
            textposition='auto'
        ))

    is_percentage = any(stat.endswith(s) for s in ['%', 'Percentage', 'Rate'])

    fig.update_layout(
        title=f"{stat_min_max.capitalize()} {stat} per Year for {data_type}s (Available: {first_year_available}-{last_year_available})",
        xaxis_title="Year",
        yaxis_title=stat,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="black"
        ),
        xaxis=dict(type='category', categoryorder='category ascending'),
        legend_title_text='Team',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        yaxis=dict(tickformat='.1%' if is_percentage else None)
    )

    if adjusted_start_year != year_range[0] or adjusted_end_year != year_range[1]:
        st.info(f"Showing data for years {adjusted_start_year}-{adjusted_end_year} due to data availability.")

    st.plotly_chart(fig)

def create_hover_text(row, stat, hover_data, df):
    year = row['year']
    team = row['Team']
    stat_value = row[stat]
    
    # Find the player with the min/max stat value for this year and team
    player_data = df[(df['year'] == year) & (df['Team'] == team) & (df[stat] == stat_value)].iloc[0]
    
    hover_text = f"Name: {player_data['Name']}<br>"
    hover_text += f"Year: {year}<br>"
    hover_text += f"Team: {team}<br>"
    hover_text += f"{stat}: {format_value(stat_value, stat)}<br>"
    
    for key in hover_data:
        if key in player_data and key != stat:
            value = player_data[key]
            hover_text += f"{key}: {format_value(value, key)}<br>"
    
    return hover_text.rstrip('<br>')

def league_wide_stats_view():
    st.subheader("Historical Histogram")

    st.markdown("""
    This tool generates a historical view of league-wide statistics for either hitters or pitchers. Here's how it works:

    1. Choose between hitters or pitchers statistics.
    2. Select a year range for the analysis.
    3. Optionally, set minimum playing time filters to focus on players with significant playing time.
    4. Choose a specific statistic to analyze.
    5. Decide whether to plot the maximum or minimum value for each year.
    6. The tool will generate a histogram showing how the chosen statistic has changed over time, highlighting the best (or worst) performer each year.

    Key features:
    - Minimum playing time filters allow you to exclude players with limited appearances.
    - The hover information provides additional stats for the highlighted player each year.
    - You can easily compare how top performances in a particular stat have evolved over time.

    This visualization is great for identifying historical trends, standout seasons, and how the boundaries of performance have shifted over time. Remember that changes in league conditions, rules, and other factors can influence these trends beyond just player performance.
    """)
    
    player_type = st.radio("Would you like to see stats for hitters or pitchers?", ("Hitters", "Pitchers"))
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"
    
    data_df = load_and_filter_data(data_type)
    
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Add minimum filter options
    use_min_filter = st.checkbox("Set minimum playing time filter?")
    min_games = min_pa = min_ip = None
    if use_min_filter:
        col1, col2 = st.columns(2)
        with col1:
            min_games = st.number_input("Minimum games per season:", min_value=1, value=20)
        with col2:
            if data_type == "Hitter":
                min_pa = st.number_input("Minimum PA per season:", min_value=1, value=200)
            else:  # Pitcher
                min_ip = st.number_input("Minimum IP per season:", min_value=1, value=50)
    
    if data_type == "Pitcher":
        default_stat = 'ERA'
        available_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP', 'WAR', 'W', 'L', 'SV', 'G', 'GS', 'IP', 'H', 'R', 'ER', 'HR', 'SO', 'AVG', 'BABIP', 'LOB%', 'GB%', 'HR/FB', 'WAR']
    else:  # Hitter
        default_stat = 'HR'
        available_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+', 'WAR', 'HR', 'R', 'RBI', 'SB', 'BB%', 'K%', 'ISO', 'BABIP', 'GB%', 'FB%', 'Hard%']
    
    available_stats = [stat for stat in available_stats if stat in data_df.columns]
    
    default_index = available_stats.index(default_stat) if default_stat in available_stats else 0
    stat = st.selectbox("Select the stat you want to plot", available_stats, index=default_index)
    
    stat_min_max = st.radio("Plot max or min of the stat?", ('max', 'min'))
    
    if data_type == "Pitcher":
        hover_data = ['W', 'L', 'ERA', 'WHIP', 'K/9', 'BB/9', 'FIP', 'WAR']
    else:  # Hitter
        hover_data = ['AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'SB', 'WAR']
    
    hover_data = [stat for stat in hover_data if stat in data_df.columns]
    
    if st.button("Generate Histogram"):
        # Apply minimum filters if specified
        if use_min_filter:
            # Filter by games and PA/IP for each season
            data_df = data_df.groupby(['IDfg', 'year']).filter(
                lambda x: x['G'].iloc[0] >= min_games and 
                (x['PA'].iloc[0] >= min_pa if data_type == "Hitter" else x['IP'].iloc[0] >= min_ip)
            )
        
        plot_league_wide_stat(data_df, stat, year_range, stat_min_max, hover_data, data_type)