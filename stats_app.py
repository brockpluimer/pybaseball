import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize



# Load team colors from the text file
@st.cache_data
def load_team_colors():
    team_colors = {}
    with open('team_colors.txt', 'r') as file:
        for line in file:
            team, color = line.strip().split(': ')
            team_colors[team.upper()] = color
    return team_colors

@st.cache_data
def load_and_filter_data(data_type, player_names_or_ids=None):
    data_dir = 'hitter_data' if data_type == "Hitter" else 'pitcher_data'
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            year = filename.split('_')[-1].split('.')[0]
            file_path = os.path.join(data_dir, filename)
            data = pd.read_csv(file_path)
            data['year'] = int(year)
            data['player_type'] = data_type.lower()
            if player_names_or_ids:
                player_data = data[
                    data['Name'].isin(player_names_or_ids) | 
                    data['IDfg'].astype(str).isin(player_names_or_ids)
                ]
                if not player_data.empty:
                    all_data.append(player_data)
            else:
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def get_team_color(team, team_colors):
    return team_colors.get(str(team).upper(), 'grey')

def display_player_stats(player_data, player_type):
    team_colors = load_team_colors()
    id_to_name = player_data.groupby('IDfg')['Name'].first().to_dict()
    player_data = player_data.sort_values(['IDfg', 'year'])

    # Define stat order for pitchers and hitters
    pitcher_stat_order = ['WAR', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'AVG' 'WHIP', 'FIP', 'CG', 'ShO', 'SV', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'BS', 'TBF', 'H', 'R', 'HR', 'SO', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'GB%', 'FB%', 'LD%', 'IFH', 'IFFB', 'Balls', 'Strikes', 'Pitches']
    hitter_stat_order = ['WAR', 'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'AVG', 'OBP', 'SLG', 'OPS', 'BB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'GB', 'FB', 'BB%', 'K%', 'BB/K', 'ISO']

    stat_order = pitcher_stat_order if player_type == "Pitcher" else hitter_stat_order

    # Define rate stats (stats that should be averaged instead of summed)
    rate_stats = ['AVG', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO', 'BABIP', 'wRC+', 'wOBA', 'WAR', 'ERA', 'WHIP', 'K/9', 'BB/9', 'H/9','HR/9', 'K/BB', 'FIP', 'xFIP']

    st.header("Career Summary")
    for idfg in player_data['IDfg'].unique():
        player_career = player_data[player_data['IDfg'] == idfg]
        player_name = id_to_name[idfg]
        st.write(f"{player_name} (ID: {idfg}): {player_career['year'].min()} - {player_career['year'].max()} ({len(player_career)} seasons)")

    st.header("Career Stats")
    career_stats = player_data.groupby('IDfg').agg(
        {col: 'mean' if col in rate_stats else 'sum' 
         for col in player_data.select_dtypes(include=['int64', 'float64']).columns 
         if col not in ['year', 'IDfg']}
    )
    career_stats['Name'] = career_stats.index.map(id_to_name)
    
    if 'Age' in career_stats.columns:
        age_col = career_stats.pop('Age')
        career_stats['Age'] = age_col

    career_stats = career_stats.reset_index().set_index(['Name', 'IDfg'])
    
    # Reorder columns based on stat_order
    ordered_cols = [col for col in stat_order if col in career_stats.columns]
    other_cols = [col for col in career_stats.columns if col not in ordered_cols and col not in ['Name', 'IDfg']]
    career_stats = career_stats[ordered_cols + other_cols]

    # Function to format numbers while preserving original precision
    def format_number(x):
        if isinstance(x, int):
            return f"{x}"
        elif isinstance(x, float):
            if x.is_integer():
                return f"{int(x)}"
            else:
                return f"{x:.3f}"
        return x

    st.dataframe(career_stats.applymap(format_number))

    st.header("Yearly Stats")
    yearly_stats = player_data.copy()
    yearly_stats['Name'] = yearly_stats['IDfg'].map(id_to_name)
    
    # Reorder columns for yearly stats
    ordered_cols = ['Name', 'IDfg', 'year'] + [col for col in stat_order if col in yearly_stats.columns]
    other_cols = [col for col in yearly_stats.columns if col not in ordered_cols]
    yearly_stats = yearly_stats[ordered_cols + other_cols]
    
    formatted_yearly_stats = yearly_stats.applymap(format_number)
    st.dataframe(formatted_yearly_stats.set_index(['Name', 'IDfg', 'year']))

    st.header("Stat Explorer")
    numeric_columns = player_data.select_dtypes(include=['int64', 'float64']).columns
    stat_options = [col for col in numeric_columns if col not in ['year', 'IDfg']]
    
    # Determine default stat based on player type
    if player_type == "Pitcher":
        default_stat = 'ERA' if 'ERA' in stat_options else stat_options[0]
    else:
        default_stat = 'HR' if 'HR' in stat_options else stat_options[0]

    selected_stat = st.selectbox("Choose a stat to visualize:", stat_options, index=stat_options.index(default_stat))

    # Season-by-season plot
    fig = px.line(player_data, x='year', y=selected_stat, color='IDfg', 
                  title=f"{selected_stat} Over Time",
                  hover_data={'IDfg': False, 'Name': True, selected_stat: ':.2f'})

    fig.update_traces(hovertemplate='Name: %{customdata[0]}<br>Year: %{x}<br>' + f'{selected_stat}: ' + '%{y:.2f}<extra></extra>')

    for trace in fig.data:
        idfg = int(trace.name)
        player_subset = player_data[player_data['IDfg'] == idfg]
        if not player_subset.empty:
            player_info = player_subset.iloc[-1]
            trace.name = player_info['Name']
            trace.line.color = get_team_color(player_info['Team'], team_colors)
            trace.customdata = player_subset[['Name']]
        else:
            trace.name = f"Unknown Player (ID: {idfg})"
            trace.line.color = 'grey'
            trace.customdata = np.full((len(trace.x), 1), f"Unknown Player (ID: {idfg})")
    
    st.plotly_chart(fig)

    # Calculate cumulative stats or career average for rate stats
    career_data = []
    for idfg, group in player_data.groupby('IDfg'):
        group = group.sort_values('year')
        group['career_year'] = range(1, len(group) + 1)
        
        if selected_stat in rate_stats:
            if player_type == "Pitcher":
                weight = 'IP'
            else:  # Hitter
                weight = 'PA' if 'PA' in group.columns else 'G'
            
            group[f'Career_Avg_{selected_stat}'] = (group[selected_stat] * group[weight]).cumsum() / group[weight].cumsum()
            title = f"Weighted Career Average {selected_stat} Over Time"
            y_axis = f'Career_Avg_{selected_stat}'
        else:
            group[f'Cumulative_{selected_stat}'] = group[selected_stat].cumsum()
            title = f"Cumulative {selected_stat} Over Career Years"
            y_axis = f'Cumulative_{selected_stat}'
        
        career_data.append(group)

    career_df = pd.concat(career_data)

    # Create the figure with custom hover template
    fig_career = px.line(career_df, x='career_year', y=y_axis, 
                         color='IDfg',
                         title=title,
                         labels={'career_year': 'Career Year'},
                         hover_data={'IDfg': False, 'Name': True, y_axis: ':.2f'})

    # Update hover template to show player name
    fig_career.update_traces(hovertemplate='Name: %{customdata[0]}<br>Career Year: %{x}<br>' + f'{y_axis}: ' + '%{y:.2f}<extra></extra>')

    # Customize the lines for each player
    for trace in fig_career.data:
        idfg = int(trace.name)
        player_subset = career_df[career_df['IDfg'] == idfg]
        if not player_subset.empty:
            player_info = player_subset.iloc[-1]
            trace.name = player_info['Name']
            trace.line.color = get_team_color(player_info['Team'], team_colors)
            # Add custom data for hover
            trace.customdata = player_subset[['Name']]
        else:
            trace.name = f"Unknown Player (ID: {idfg})"
            trace.line.color = 'grey'
            trace.customdata = np.full((len(trace.x), 1), f"Unknown Player (ID: {idfg})")

    st.plotly_chart(fig_career)

    if selected_stat in rate_stats:
        st.info(f"{selected_stat} is a rate stat, so the second graph shows the career average at each point in time.")
    else:
        st.info(f"{selected_stat} is a counting stat, so the second graph shows cumulative values over time.")

def individual_player_view():
    st.subheader("Individual Player Statistics")

    player_type = st.radio("Would you like to see stats for a hitter or a pitcher?", ("Hitter", "Pitcher"))

    default_player = "Clayton Kershaw" if player_type == "Pitcher" else "Shohei Ohtani"
    player_input = st.text_input("Enter player name or FanGraphs ID:", default_player)
    
    if st.button("Load Player Data") or ('player_data' in st.session_state and st.session_state.player_data is not None):
        if 'player_data' not in st.session_state or st.session_state.player_data is None:
            hitter_data = load_and_filter_data("Hitter", [player_input])
            pitcher_data = load_and_filter_data("Pitcher", [player_input])
            st.session_state.player_data = pd.concat([hitter_data, pitcher_data])
        
        if st.session_state.player_data.empty:
            st.error(f"No data found for {player_input}")
        else:
            st.success(f"Data loaded for {player_input}")
            
            filtered_data = st.session_state.player_data[st.session_state.player_data['player_type'] == player_type.lower()]
            
            if filtered_data.empty:
                st.warning(f"No {player_type.lower()} data found for {player_input}. They might be a {['pitcher', 'hitter'][player_type == 'Hitter']}.")
                if st.button(f"Show {['pitcher', 'hitter'][player_type == 'Hitter']} data instead"):
                    filtered_data = st.session_state.player_data[st.session_state.player_data['player_type'] != player_type.lower()]
            
            if not filtered_data.empty:
                display_player_stats(filtered_data, player_type)
            else:
                st.error(f"No data available for {player_input}")

def compare_players_view():
    st.subheader("Compare Players")

    player_type = st.radio("Would you like to compare hitters or pitchers?", ("Hitters", "Pitchers"))
    data_type = "Pitcher" if player_type == "Pitchers" else "Hitter"

    st.subheader("Enter up to 10 player names or FanGraphs IDs (one per line):")
    default_players = "Clayton Kershaw\nSandy Koufax" if data_type == "Pitcher" else "Shohei Ohtani\nMookie Betts"
    player_inputs = st.text_area("Player Names or IDs", default_players).split('\n')
    player_inputs = [input.strip() for input in player_inputs if input.strip()][:10]  # Limit to 10 players

    if 'player_data' in st.session_state:
        if st.session_state.player_data is None or (isinstance(st.session_state.player_data, pd.DataFrame) and len(st.session_state.player_data['IDfg'].unique()) == 1):
            st.session_state.player_data = None

    if st.button("Load Players Data") or ('player_data' in st.session_state and st.session_state.player_data is not None):
        if 'player_data' not in st.session_state or st.session_state.player_data is None:
            hitter_data = load_and_filter_data("Hitter", player_inputs)
            pitcher_data = load_and_filter_data("Pitcher", player_inputs)
            st.session_state.player_data = pd.concat([hitter_data, pitcher_data])

        if st.session_state.player_data.empty:
            st.error("No data found for the specified players")
        else:
            filtered_data = st.session_state.player_data[st.session_state.player_data['player_type'] == data_type.lower()]
            
            if filtered_data.empty:
                st.warning(f"No {data_type.lower()} data found for the specified players. They might be {['pitchers', 'hitters'][data_type == 'Hitter']}.")
                if st.button(f"Show {['pitcher', 'hitter'][data_type == 'Hitter']} data instead"):
                    filtered_data = st.session_state.player_data[st.session_state.player_data['player_type'] != data_type.lower()]
            
            if not filtered_data.empty:
                st.success(f"Data loaded for {len(filtered_data['IDfg'].unique())} player(s)")
                display_player_stats(filtered_data, data_type)
            else:
                st.error("No data available for the specified players")

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

def calculate_similarity_scores(player_data, target_player, stats_to_compare, mode='season', scaling_factor=10):
    # Filter players of the same type (hitter or pitcher)
    player_data = player_data[player_data['player_type'] == target_player['player_type']]
    
    if mode == 'season':
        # For season mode, we'll compare individual seasons
        # Exclude the target player's other seasons
        players_stats = player_data[
            (player_data['IDfg'] != target_player['IDfg']) | 
            (player_data['year'] == target_player['year'])
        ]
        players_stats = players_stats[stats_to_compare + ['IDfg', 'Name', 'year']]
    else:  # Career mode
        # For career mode, we'll compare career averages
        numeric_stats = [stat for stat in stats_to_compare if player_data[stat].dtype in ['int64', 'float64']]
        players_stats = player_data.groupby('IDfg').agg({
            **{stat: 'mean' for stat in numeric_stats},
            'Name': 'first',
            'year': ['min', 'max']
        }).reset_index()
        
        players_stats.columns = ['IDfg'] + numeric_stats + ['Name', 'First Year', 'Last Year']
        players_stats['Years'] = players_stats['Last Year'] - players_stats['First Year'] + 1
    
    # Remove players with missing data for any of the selected stats
    players_stats = players_stats.dropna(subset=stats_to_compare)
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_stats = scaler.fit_transform(players_stats[stats_to_compare])
    
    # Calculate distances
    target_player_stats = normalized_stats[players_stats['IDfg'] == target_player['IDfg']]
    distances = euclidean_distances(target_player_stats, normalized_stats)[0]
    
    # Create similarity scores with scaling factor
    similarity_scores = np.exp(-scaling_factor * distances)
    
    # Create a dataframe with results
    results = pd.DataFrame({
        'IDfg': players_stats['IDfg'],
        'Name': players_stats['Name'],
        'Similarity': similarity_scores
    })
    
    # Add year information
    if mode == 'season':
        results['year'] = players_stats['year']
    else:
        results['Years'] = players_stats['Years']
        results['First Year'] = players_stats['First Year']
        results['Last Year'] = players_stats['Last Year']
    
    # Add all stats for hover data
    for stat in stats_to_compare:
        results[stat] = players_stats[stat]
    
    # Sort by similarity and exclude the target player's season
    if mode == 'season':
        results = results[
            (results['IDfg'] != target_player['IDfg']) | 
            (results['year'] != target_player['year'])
        ].sort_values('Similarity', ascending=False)
    else:
        results = results[results['IDfg'] != target_player['IDfg']].sort_values('Similarity', ascending=False)
    
    return results

def player_similarity_view():
    st.subheader("Player Similarity Scores")

    st.markdown("""
    This tool finds players who are most similar to a selected player based on chosen statistical categories. Here's how it works:

    1. Select whether you want to compare hitters or pitchers.
    2. Choose if you want to compare individual seasons or entire careers.
    3. Choose a specific player to analyze.
    4. Decide how many similar players you want to find.
    5. Select the statistical categories you want to use for comparison. Default categories are provided, but you can customize these.
    6. The tool will then calculate similarity scores based on these stats and show you the most similar players.
    7. A scatter plot will be displayed, comparing the selected stat and similarity scores of similar players to your chosen player.

    This analysis uses a mathematical approach to find similarities and doesn't account for era differences, park factors, or other contextual elements. It's a fun way to explore player comparisons but should not be considered a definitive measure of player similarity.
    """)

    player_type = st.radio("Would you like to find similar hitters or pitchers?", ("Hitters", "Pitchers"), key="player_type_radio")
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    mode = st.radio("Select comparison mode:", ("Season", "Career"), key="comparison_mode_radio")

    all_data = load_and_filter_data(data_type)
    players = all_data[['IDfg', 'Name', 'player_type']].drop_duplicates()
    
    # Set Clayton Kershaw as the default pitcher and Shohei Ohtani as the default hitter
    default_player = "Clayton Kershaw" if data_type == "Pitcher" else "Shohei Ohtani"
    
    # Check if the default player is in the dataset, if not use the first player in the list
    if default_player not in players['Name'].unique():
        default_player = players['Name'].iloc[0]
    
    target_player_name = st.selectbox(
        f"Select a {player_type.lower()[:-1]}:",
        players['Name'].unique(),
        index=list(players['Name'].unique()).index(default_player),
        key="player_name_selectbox"
    )
    
    if mode == "Season":
        seasons = sorted(all_data[all_data['Name'] == target_player_name]['year'].unique(), reverse=True)
        target_year = st.selectbox(
            "Select season:",
            seasons,
            index=0,
            key="season_selectbox"
        )
        target_player = all_data[(all_data['Name'] == target_player_name) & (all_data['year'] == target_year)].iloc[0]
    else:
        target_player = players[players['Name'] == target_player_name].iloc[0]

    num_similar_players = st.slider("Number of similar players to find:", 1, 20, 5, key="num_similar_players_slider")
    
    if data_type == "Pitcher":
        default_stats = ['WAR', 'ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP']
    else:
        default_stats = ['WAR', 'AVG', 'OBP', 'SLG', 'wRC+', 'ISO', 'BB%', 'K%']
    
    available_stats = [col for col in all_data.columns if col not in ['IDfg', 'Name', 'Team', 'year', 'player_type']]
    stats_to_compare = st.multiselect(
        "Select stats to compare:",
        available_stats,
        default=default_stats,
        key="stats_to_compare_multiselect"
    )

    scaling_factor = st.slider(
        "Similarity scaling factor:",
        1, 50, 10,
        help="Higher values make similarity scores more sensitive to differences",
        key="scaling_factor_slider"
    )

    if st.button("Find Similar Players"):
        similarity_scores = calculate_similarity_scores(all_data, target_player, stats_to_compare, mode.lower(), scaling_factor)
        
        if similarity_scores.empty:
            st.warning(f"No similar players found for {target_player_name} using the selected stats. This may be due to missing data for the selected player or stats. Try selecting different stats.")
        else:
            st.subheader(f"Players most similar to {target_player_name}")
            for _, player in similarity_scores.head(num_similar_players).iterrows():
                player_name = f"{player['Name']} ({player['year']})" if mode == "Season" else f"{player['Name']} ({player['First Year']}-{player['Last Year']})"
                st.write(f"{player_name} (Similarity: {player['Similarity']:.2f})")
            
            # Determine which stat to use for the y-axis
            y_stat = 'WAR' if 'WAR' in similarity_scores.columns else stats_to_compare[0]
            
            # Prepare hover data
            hover_data = ['Name', 'year' if mode == 'Season' else 'Years', 'Similarity'] + stats_to_compare

            # Create the scatter plot
            fig = px.scatter(
                similarity_scores.head(num_similar_players),
                x='Similarity',
                y=y_stat,
                hover_name='Name',
                hover_data=hover_data,
                title=f"Top {num_similar_players} Similar Players to {target_player_name} ({mode} Comparison)"
            )

            # Add target player as a different marker
            target_data = pd.DataFrame([target_player])
            if y_stat not in target_data.columns:
                # Use the mean of the stat for the target player if not available
                target_data[y_stat] = all_data[all_data['IDfg'] == target_player['IDfg']][y_stat].mean()
            
            fig.add_trace(px.scatter(
                target_data,
                x=[1],  # Maximum similarity
                y=[target_data[y_stat].iloc[0]],
                hover_name='Name',
                hover_data=[col for col in hover_data if col in target_data.columns],
                color_discrete_sequence=['red']
            ).data[0])

            # Customize the layout
            fig.update_layout(
                xaxis_title="Similarity Score",
                yaxis_title=y_stat,
                showlegend=False
            )

            # Display the plot
            st.plotly_chart(fig)

    st.info(f"Note: This similarity comparison is for {player_type.lower()} only. To find similar {'pitchers' if player_type == 'Hitters' else 'hitters'}, please start a new similarity search.")

def custom_war_generator():
    st.subheader("Custom WAR Generator")

    # Explanation of WAR
    st.markdown("""
    ### What is WAR?
    WAR (Wins Above Replacement) is a comprehensive statistic that attempts to summarize a player's total contributions to their team in one statistic. It's calculated differently for position players and pitchers. As a general rule, a player who earns 0-1 WAR in a year is considered replacement level; 1-2 is a utility player, 2-3 is an average starter, 3-4 is an above average starter, 4-5 is an All-Star, 5-6 is a Superstar, and 6+ is MVP-level

    #### bWAR (Baseball-Reference WAR):
    - For position players: Combines batting, baserunning, and fielding value adjusted for position and league.
    - For pitchers: Based on runs allowed with adjustments for team defense, ballpark, and quality of opposition.

    #### fWAR (FanGraphs WAR):
    - For position players: Similar to bWAR but uses different defensive metrics and offensive weights.
    - For pitchers: Based on FIP (Fielding Independent Pitching) rather than runs allowed.

    Both versions aim to measure a player's value in terms of wins above what a replacement-level player would provide.

    This tool allows you to create your own version of WAR by adjusting the weights of various statistics. The point is, what do YOU think makes a player valuable? Note that the maximum WAR a player can earn in this model is 15, and the lowest is -5.
    
    ### Understanding the Weight Range (-1 to 1)
    
    The weight range from -1 to 1 determines how much each statistic contributes to the custom WAR calculation:
    
    - A weight of 1 means the stat has the maximum positive impact on the WAR calculation.
    - A weight of -1 means the stat has the maximum negative impact on the WAR calculation.
    - A weight of 0 means the stat has no impact on the WAR calculation.
    - Values between 0 and 1 (or 0 and -1) represent varying degrees of positive (or negative) impact.
    
    For example, if you set the weight for home runs (HR) to 0.5 and the weight for strikeouts (SO) to -0.3, 
    it means that home runs will have a moderate positive impact on a player's WAR, while strikeouts will have 
    a slightly smaller negative impact. The relative magnitudes of these weights determine how much each stat 
    influences the final WAR value.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"))
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    data_df = load_and_filter_data(data_type)

    calculation_type = st.radio("Select calculation type:", ("Season-by-Season", "Career Average"))

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

    if data_type == "Hitter":
        default_stats = ['AVG', 'OBP', 'SLG', 'HR', 'RBI', 'SB', 'BB%', 'K%', 'ISO', 'wRC+', 'Def']
    else:  # Pitcher
        default_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP', 'IP', 'W', 'SV']

    additional_stats = st.multiselect("Select additional stats to include:", 
                                      [col for col in data_df.columns if col not in default_stats + ['Name', 'Team', 'year', 'Age', 'IDfg']])
    
    all_stats = default_stats + additional_stats

    stat_toggles = {}
    for stat in all_stats:
        if stat in data_df.columns:
            stat_toggles[stat] = st.checkbox(f"Include {stat}", value=True)

    stats_to_use = [stat for stat in all_stats if stat_toggles.get(stat, False)]

    weights = {}
    for stat in stats_to_use:
        weight = st.number_input(f"Weight for {stat}", min_value=-1.0, max_value=1.0, value=0.1, step=0.0001, format="%.4f")
        weights[stat] = weight

    if st.button("Generate Custom WAR"):
        if not stats_to_use:
            st.warning("Please select at least one stat to include in the WAR calculation.")
            return

        if use_min_filter:
            if data_type == "Hitter":
                data_df = data_df[(data_df['G'] >= min_games) & (data_df['PA'] >= min_pa)]
            else:  # Pitcher
                data_df = data_df[(data_df['G'] >= min_games) & (data_df['IP'] >= min_ip)]

        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data_df[stats_to_use]), 
                                   columns=stats_to_use, index=data_df.index)

        custom_war = pd.Series(0, index=data_df.index)
        for stat, weight in weights.items():
            custom_war += scaled_data[stat] * weight

        custom_war = (custom_war - custom_war.min()) / (custom_war.max() - custom_war.min()) * 20 - 5

        data_df['Custom WAR'] = custom_war

        if calculation_type == "Career Average":
            career_war = data_df.groupby('Name').agg({
                'Custom WAR': 'mean',
                'year': ['min', 'max', 'count']
            }).reset_index()
            career_war.columns = ['Name', 'Custom WAR', 'First Year', 'Last Year', 'Years Played']
            data_df = career_war
        else:
            data_df['Years Played'] = 1
            data_df['First Year'] = data_df['year']
            data_df['Last Year'] = data_df['year']

        st.session_state.custom_war_data = data_df
        st.session_state.calculation_type = calculation_type

        st.success("Custom WAR calculated successfully. Use the date range slider below to view results.")

    if 'custom_war_data' in st.session_state:
        data_df = st.session_state.custom_war_data
        calculation_type = st.session_state.calculation_type

        if calculation_type == "Season-by-Season":
            min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
            top_players_start_year, top_players_end_year = st.slider(
                "Select the year range for top players:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            filtered_df = data_df[(data_df['year'] >= top_players_start_year) & (data_df['year'] <= top_players_end_year)]
        else:
            filtered_df = data_df

        st.subheader(f"Top Players by Custom WAR ({calculation_type})")
        top_players = filtered_df.sort_values('Custom WAR', ascending=False).head(20)
        st.table(top_players[['Name', 'Custom WAR', 'Years Played', 'First Year', 'Last Year']].round(2))

        fig = px.histogram(filtered_df, x='Custom WAR', nbins=50, 
                           title=f"Distribution of Custom WAR ({calculation_type})")
        st.plotly_chart(fig)

        if 'WAR' in filtered_df.columns:
            fig = px.scatter(filtered_df, x='WAR', y='Custom WAR', hover_data=['Name', 'year'],
                             title=f"Custom WAR vs Traditional WAR ({calculation_type})")
            st.plotly_chart(fig)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Custom WAR Data",
            data=csv,
            file_name=f"custom_war_data_{calculation_type.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

def how_is_he_the_goat():
    st.subheader("How is he the GOAT?")

    st.markdown("""
    This tool attempts to find the optimal weights for a selected set of baseball statistics that would make a chosen player the Greatest of All Time (GOAT). Here's how it works:
    
    1. You select a player and a time frame.
    2. The tool considers a predefined set of key stats for batters or pitchers.
    3. It then calculates weights for these stats that would rank your chosen player as the best among all players.
    4. If successful, it shows the weights and the resulting player rankings.
    5. If unsuccessful, it informs you that it's impossible to make that player the GOAT with the given data.

    Note: Some statistics are inversely weighted in the calculation. For batters, this includes stats like Strikeouts (SO) and Ground into Double Play (GDP), where lower values are better. For pitchers, this includes ERA, Walks (BB), and Hits Allowed (H), among others. The tool automatically adjusts for these "negative" stats in its calculations.


    This is a mathematical exercise and doesn't necessarily reflect real-world value. It's designed to explore what aspects of a player's performance would need to be emphasized to consider them the greatest.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"), key="goat_player_type_radio")
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    data_df = load_and_filter_data(data_type)

    # Year range selection
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year), key="goat_year_range_slider")
    data_df = data_df[(data_df['year'] >= year_range[0]) & (data_df['year'] <= year_range[1])]

    # Player selection with defaults
    players = data_df['Name'].unique()
    default_player = "Shohei Ohtani" if data_type == "Hitter" else "Clayton Kershaw"
    
    # Check if the default player is in the dataset, if not use the first player in the list
    if default_player not in players:
        default_player = players[0]
    
    selected_player = st.selectbox(
        "Select a player:",
        players,
        index=list(players).index(default_player),
        key="goat_player_selectbox"
    )

    # Define stats to use based on player type
    if data_type == "Hitter":
        stats_to_use = ['G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'AVG', 'GB', 'FB', 'LD', 'IFFB', 'Pitches', 'Balls', 'Strikes', 'IFH', 'BB%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'Bat', 'Fld', 'Spd']
        # Stats where higher is worse
        negative_stats = ['GDP', 'CS', 'SO', 'K%']
    else:  # Pitcher
        stats_to_use = ['W', 'L', 'ERA', 'G', 'GS', 'CG', 'ShO', 'SV', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'SO', 'GB', 'FB', 'LD', 'IFFB', 'Balls', 'Strikes', 'Pitches', 'RS', 'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'LOB%', 'FIP']
        # Stats where higher is worse
        negative_stats = ['L', 'ERA', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'BB/9', 'H/9', 'HR/9', 'AVG', 'WHIP', 'BABIP', 'FIP']

    # Filter stats that are actually in the dataframe
    stats_to_use = [stat for stat in stats_to_use if stat in data_df.columns]

    if st.button("Find GOAT Weights"):
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data_df[stats_to_use]), 
                                   columns=stats_to_use, index=data_df.index)

        # Invert negative stats
        for stat in negative_stats:
            if stat in scaled_data.columns:
                scaled_data[stat] = 1 - scaled_data[stat]

        # Group by player and take the mean of their stats
        player_stats = scaled_data.groupby(data_df['Name']).mean()

        # Handle missing data for the selected player
        selected_player_stats = player_stats.loc[selected_player]
        available_stats = selected_player_stats.dropna().index
        player_stats = player_stats[available_stats]

        # Function to optimize
        def objective(weights):
            war = (player_stats * weights).sum(axis=1)
            player_war = war[selected_player]
            return np.sum(np.maximum(0, war - player_war))  # Sum of how much others exceed the player

        # Constraints: sum of absolute weights = 1
        def constraint(weights):
            return np.sum(np.abs(weights)) - 1

        # Initial guess
        initial_weights = np.ones(len(available_stats)) / len(available_stats)

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                          constraints={'type': 'eq', 'fun': constraint},
                          options={'ftol': 1e-10, 'maxiter': 1000})

        if result.success and result.fun < 1e-5:  # Check if optimization succeeded and player is at the top
            optimal_weights = result.x
            
            # Calculate WAR with optimal weights
            war = (player_stats * optimal_weights).sum(axis=1)
            player_rankings = war.sort_values(ascending=False)
            
            st.success(f"Optimal weights found to make {selected_player} the GOAT!")
            
            # Display weights in ranked order
            st.subheader("Optimal Weights (Ranked by Importance):")
            weight_df = pd.DataFrame({'Stat': available_stats, 'Weight': optimal_weights})
            weight_df['Abs_Weight'] = np.abs(weight_df['Weight'])
            weight_df = weight_df.sort_values('Abs_Weight', ascending=False)
            for _, row in weight_df.iterrows():
                if row['Abs_Weight'] > 1e-4:  # Only show weights that are not essentially zero
                    st.write(f"{row['Stat']}: {row['Weight']:.4f}")
            
            # Display player ranking
            st.subheader("Top 10 Players with these weights:")
            st.table(player_rankings.head(10))
            
            # Show selected player's rank
            player_rank = player_rankings.index.get_loc(selected_player) + 1
            st.write(f"{selected_player}'s rank with these weights: {player_rank}")

            # Generate explanation
            top_positive_stats = weight_df[weight_df['Weight'] > 0].nlargest(3, 'Abs_Weight')['Stat'].tolist()
            top_negative_stats = weight_df[weight_df['Weight'] < 0].nlargest(2, 'Abs_Weight')['Stat'].tolist()
            
            explanation = f"{selected_player} is considered the GOAT in this analysis primarily due to "
            explanation += f"their exceptional performance in {', '.join(top_positive_stats[:-1])}, and {top_positive_stats[-1]}. "
            if top_negative_stats:
                explanation += f"The model also values their ability to minimize {' and '.join(top_negative_stats)}. "
            explanation += "This combination of strengths sets them apart in this particular weighting scheme."
            
            st.subheader("Explanation:")
            st.write(explanation)
            
            # Display stats used
            st.subheader("Stats Used in Calculation:")
            st.write(", ".join(available_stats))
            
        else:
            st.error(f"Sadly, it is impossible for {selected_player} to be the GOAT with the given stats and time frame.")
            st.write("Try adjusting the year range or selecting a different player.")

def main():
    st.title("Brock's Baseball Stats Explorer")
    
    st.header("It's Time For Dodger Baseball! (or your team who sucks)")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the mode",
                                ["Individual Player", "Compare Players", "Historical Histogram", 
                                 "Career Stat Race", "Player Similarity", "Custom WAR Generator",
                                 "How is he the GOAT?"])
    
    # Reset session state when changing modes
    if 'current_mode' not in st.session_state or st.session_state.current_mode != app_mode:
        st.session_state.player_data = None
        st.session_state.current_mode = app_mode
    
    if app_mode == "Individual Player":
        individual_player_view()
    elif app_mode == "Compare Players":
        compare_players_view()
    elif app_mode == "Historical Histogram":
        league_wide_stats_view()
    elif app_mode == "Career Stat Race":
        race_chart_view()
    elif app_mode == "Player Similarity":
        player_similarity_view()
    elif app_mode == "Custom WAR Generator":
        custom_war_generator()
    else:  # How is he the GOAT?
        how_is_he_the_goat()

if __name__ == "__main__":
    main()
