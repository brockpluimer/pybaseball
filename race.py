import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional
from load_data import load_and_filter_data
from colors import load_team_colors, get_team_color

def process_data_for_race(df, stat, start_year, end_year, player_type, race_type, min_games=None):
    # Check if the stat exists in the dataframe
    if stat not in df.columns:
        raise ValueError(f"The stat '{stat}' is not available in the dataset.")

    # Find the first and last years where the stat is available
    valid_years = df[df[stat].notnull()]['year']
    if valid_years.empty:
        raise ValueError(f"No valid data available for the stat '{stat}'.")

    first_year_available = valid_years.min()
    last_year_available = valid_years.max()

    # Adjust the year range if necessary
    adjusted_start_year = max(start_year, first_year_available)
    adjusted_end_year = min(end_year, last_year_available)

    if adjusted_start_year > adjusted_end_year:
        raise ValueError(f"No data available for '{stat}' in the selected year range.")

    # Filter the dataframe for the adjusted year range
    df = df[(df['year'] >= adjusted_start_year) & (df['year'] <= adjusted_end_year)]

    # Define rate stats for hitters and pitchers
    hitter_rate_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO', 'BABIP', 'wRC+', 'wOBA', 'Off', 'Def', 'BsR', 'RAR', 'WAR/162', 'Off/162', 'Def/162', 'WPA', '-WPA', '+WPA', 'RE24', 'REW', 'pLI', 'phLI', 'WPA/LI', 'Clutch', 'FB%', 'GB%', 'LD%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'IFFB%', 'IFH%', 'BUH%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%']
    pitcher_rate_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'K%', 'BB%', 'K-BB%', 'AVG', 'BABIP', 'LOB%', 'FIP', 'xFIP', 'ERA-', 'FIP-', 'xFIP-', 'WPA', '-WPA', '+WPA', 'RE24', 'REW', 'pLI', 'inLI', 'gmLI', 'WPA/LI', 'Clutch', 'FB%', 'GB%', 'LD%', 'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%']
    
    rate_stats = hitter_rate_stats if player_type == "Hitter" else pitcher_rate_stats
    
    # Apply minimum games filter if specified
    if min_games is not None:
        # Calculate average games per season for each player
        avg_games = df.groupby('IDfg')['G'].mean()
        qualified_players = avg_games[avg_games >= min_games].index
        df = df[df['IDfg'].isin(qualified_players)]

    # Stats to display with 3 decimal points
    three_decimal_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'wOBA']
    
    if stat in rate_stats:
        # For rate stats, use weighted average
        if player_type == "Pitcher":
            weight = 'IP'
        else:  # Hitter
            weight = 'PA' if 'PA' in df.columns else 'G'
        
        def safe_average(x):
            if x[weight].sum() == 0:
                return 0
            return np.average(x[stat], weights=x[weight])
        
        player_stats = df.groupby(['IDfg', 'year']).apply(safe_average).unstack(fill_value=0)
        weight_sums = df.groupby(['IDfg', 'year'])[weight].sum().unstack(fill_value=0)
        player_stats_cumsum = (player_stats * weight_sums).cumsum(axis=1) / weight_sums.cumsum(axis=1).replace(0, np.nan)
    else:
        # For counting stats, use sum and cumulative sum
        player_stats = df.groupby(['IDfg', 'year'])[stat].sum().unstack(fill_value=0)
        player_stats_cumsum = player_stats.cumsum(axis=1)
    
    id_to_name = df.sort_values('year').groupby('IDfg').last()[['Name', 'Team']]
    
    def format_value(value, stat):
        if stat in ['AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'wOBA']:
            return f"{value:.3f}"
        elif stat in ['ERA', 'FIP', 'xFIP', 'WHIP']:
            return f"{value:.2f}"
        elif stat in ['HR', 'R', 'RBI', 'SB', 'BB', '1B', '2B', '3B' 'SO', 'H', 'G', 'GS', 'W', 'L', 'SV']:
            return f"{int(value)}"
        else:
            return f"{value:.1f}"

    data_for_animation = []
    available_years = sorted(player_stats_cumsum.columns)
    for year in available_years:
        if year < adjusted_start_year or year > adjusted_end_year:
            continue
        if race_type == 'max':
            year_data = player_stats_cumsum[year].sort_values(ascending=False).head(10)
        else:  # min
            year_data = player_stats_cumsum[year].sort_values(ascending=True).head(10)
        for rank, (idfg, value) in enumerate(year_data.items(), 1):
            if pd.isna(value):
                continue
            name = id_to_name.loc[idfg, 'Name']
            team = id_to_name.loc[idfg, 'Team']
            
            formatted_value = format_value(value, stat)
            
            data_for_animation.append({
                'Year': year,
                'IDfg': idfg,
                'Name': name,
                'Value': formatted_value,
                'Value_float': value,  # Keep the original float value for sorting
                'Rank': rank,
                'Team': team
            })
    
    return pd.DataFrame(data_for_animation), adjusted_start_year, adjusted_end_year


def create_race_plot(df, stat, start_year, end_year, race_type):
    team_colors = load_team_colors()
    color_map = {idfg: get_team_color(df[df['IDfg'] == idfg]['Team'].iloc[0], team_colors) for idfg in df['IDfg'].unique()}

    # Determine the range for the x-axis
    x_min = df['Value_float'].min()
    x_max = df['Value_float'].max()
    x_range = [x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df[df['Year'] == start_year]['Value_float'],
                y=df[df['Year'] == start_year]['Name'],
                orientation='h',
                text=df[df['Year'] == start_year]['Value'],
                texttemplate='%{text}',
                textposition='outside',
                marker=dict(color=[color_map[idfg] for idfg in df[df['Year'] == start_year]['IDfg']])
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f"Top 10 {stat} {'Maxima' if race_type == 'max' else 'Minima'} Over Time",
                font=dict(size=24)
            ),
            xaxis=dict(range=x_range, autorange=False, title=stat),
            yaxis=dict(
                range=[-0.5, 9.5],
                autorange=False,
                title='Player',
                categoryorder='array',
                categoryarray=df[df['Year'] == start_year]['Name'][::-1],
                tickfont=dict(weight='bold')
            )
        )
    )

    frames = [go.Frame(
        data=[go.Bar(
            x=df[df['Year'] == year]['Value_float'],
            y=df[df['Year'] == year]['Name'],
            orientation='h',
            text=df[df['Year'] == year]['Value'],
            texttemplate='%{text}',
            textposition='outside',
            marker=dict(color=[color_map[idfg] for idfg in df[df['Year'] == year]['IDfg']])
        )],
        layout=go.Layout(
            xaxis=dict(range=x_range),
            yaxis=dict(
                categoryorder='array',
                categoryarray=df[df['Year'] == year]['Name'][::-1],
                tickfont=dict(weight='bold')
            )
        ),
        name=str(year)
    ) for year in sorted(df['Year'].unique())]

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], "label": "Pause", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Year: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(year)],
                        {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }
                    ],
                    "label": str(year),
                    "method": "animate"
                }
                for year in sorted(df['Year'].unique())
            ]
        }]
    )

    st.plotly_chart(fig)

def race_chart_view():
    st.subheader("Career Stat Race")

    st.markdown("""
    This tool creates an animated "race" chart showing how players' career statistics have evolved over time. Here's how it works:

    1. Choose between hitter or pitcher statistics.
    2. Select a start and end year for the analysis.
    3. Pick a specific statistic to track.
    4. Decide whether to race for the maximum or minimum value of the stat.
    5. Optionally, set a minimum number of games played per season to filter out players with limited playing time.
    6. The tool will generate an animated bar chart race showing how players' career totals or averages for the chosen stat have changed year by year.

    Key features:
    - The race can show either cumulative totals (for counting stats) or career averages (for rate stats).
    - You can choose to race for the highest or lowest values, depending on the nature of the statistic.
    - The minimum games filter helps focus on players with substantial playing time.
    - The animation provides a dynamic view of how player rankings have shifted over time.

    This visualization is excellent for:
    - Tracking career milestone races (e.g., all-time home run leaders)
    - Comparing career trajectories of different players
    - Identifying periods of dominance for particular players
    - Visualizing how quickly records are approached or broken

    Remember that this tool uses career totals or averages, so players who had shorter careers but exceptional peak years might not rank as highly as those with longer careers.
    """)
    
    player_type = st.radio("Select player type:", ("Hitter", "Pitcher"))
    data_df = load_and_filter_data(player_type)  # Load all data
    
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year, value=min_year)
    end_year = st.number_input("End Year", min_value=min_year, max_value=max_year, value=max_year)
    
    if start_year >= end_year:
        st.error("Start year must be less than end year.")
        return
    
    # Define stat order for pitchers and hitters
    pitcher_stat_order = ['WAR', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'AVG', 'WHIP', 'FIP', 'CG', 'ShO', 'SV', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'BS', 'TBF', 'H', 'R', 'HR', 'SO', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'GB%', 'FB%', 'LD%', 'IFH', 'IFFB', 'Balls', 'Strikes', 'Pitches']
    hitter_stat_order = ['WAR', 'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'AVG', 'OBP', 'SLG', 'OPS', 'BB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'GB', 'FB', 'BB%', 'K%', 'BB/K', 'ISO']

    # Get all available stats from the dataframe
    available_stats = list(data_df.select_dtypes(include=[int, float]).columns)
    
    # Order the stats based on the player type
    if player_type == "Pitcher":
        ordered_stats = [stat for stat in pitcher_stat_order if stat in available_stats]
    else:
        ordered_stats = [stat for stat in hitter_stat_order if stat in available_stats]
    
    # Add any remaining stats that weren't in the predefined order
    remaining_stats = [stat for stat in available_stats if stat not in ordered_stats]
    ordered_stats.extend(remaining_stats)
    
    stat = st.selectbox("Select the stat for the race chart", ordered_stats)
    
    race_type = st.radio("Select race type:", ("max", "min"))
    
    use_min_games = st.checkbox("Set minimum number of games played per season?")
    min_games = None
    if use_min_games:
        min_games = st.number_input("Minimum average games per season:", min_value=1, value=50)
    
    if st.button("Generate Race Chart"):
        try:
            processed_data, adj_start_year, adj_end_year = process_data_for_race(
                data_df, stat, start_year, end_year, player_type, race_type, min_games
            )
            if processed_data.empty:
                st.warning(f"No data available for {stat} in the selected year range.")
            else:
                if adj_start_year != start_year or adj_end_year != end_year:
                    st.warning(f"Adjusted year range to {adj_start_year}-{adj_end_year} due to data availability.")
                
                # Convert 'Value' to float for calculations
                processed_data['Value_float'] = processed_data['Value'].astype(float)
                
                # Sort the dataframe based on race_type
                processed_data = processed_data.sort_values(['Year', 'Value_float'], 
                                                           ascending=[True, race_type == 'min'])
                
                # Ensure we have top 10 for each year
                processed_data = processed_data.groupby('Year').apply(lambda x: x.nlargest(10, 'Value_float') if race_type == 'max' else x.nsmallest(10, 'Value_float')).reset_index(drop=True)
                
                create_race_plot(processed_data, stat, adj_start_year, adj_end_year, race_type)
        except ValueError as e:
            st.error(str(e))