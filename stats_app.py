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

def individual_player_view(data_type):
    player_input = st.text_input("Enter player name or FanGraphs ID:", "Shohei Ohtani")
    if st.button("Load Player Data") or ('player_data' in st.session_state and st.session_state.player_data is not None):
        if 'player_data' not in st.session_state or st.session_state.player_data is None:
            st.session_state.player_data = load_and_filter_data(data_type, [player_input])
        
        if st.session_state.player_data.empty:
            st.error(f"No data found for {player_input}")
        else:
            st.success(f"Data loaded for {player_input}")
            display_player_stats(st.session_state.player_data)

def compare_players_view(data_type):
    st.subheader("Enter up to 10 player names or FanGraphs IDs (one per line):")
    player_inputs = st.text_area("Player Names or IDs", "Shohei Ohtani\nMike Trout").split('\n')
    player_inputs = [input.strip() for input in player_inputs if input.strip()][:10]  # Limit to 10 players

    # Reset session state if coming from individual view or if it's None
    if 'player_data' in st.session_state:
        if st.session_state.player_data is None or (isinstance(st.session_state.player_data, pd.DataFrame) and len(st.session_state.player_data['IDfg'].unique()) == 1):
            st.session_state.player_data = None

    if st.button("Load Players Data") or ('player_data' in st.session_state and st.session_state.player_data is not None):
        if 'player_data' not in st.session_state or st.session_state.player_data is None:
            st.session_state.player_data = load_and_filter_data(data_type, player_inputs)
        
        if st.session_state.player_data.empty:
            st.error("No data found for the specified players")
        else:
            st.success(f"Data loaded for {len(player_inputs)} player(s)")
            display_player_stats(st.session_state.player_data)

def display_player_stats(player_data):
    team_colors = load_team_colors()
    id_to_name = player_data.groupby('IDfg')['Name'].first().to_dict()
    player_data = player_data.sort_values(['IDfg', 'year'])

    st.header("Career Summary")
    for idfg in player_data['IDfg'].unique():
        player_career = player_data[player_data['IDfg'] == idfg]
        player_name = id_to_name[idfg]
        st.write(f"{player_name} (ID: {idfg}): {player_career['year'].min()} - {player_career['year'].max()} ({len(player_career)} seasons)")

    st.header("Career Stats")
    career_stats = player_data.groupby('IDfg').agg({col: 'sum' for col in player_data.select_dtypes(include=['int64', 'float64']).columns if col not in ['year', 'IDfg']})
    career_stats['Name'] = career_stats.index.map(id_to_name)
    career_stats = career_stats.reset_index().set_index(['Name', 'IDfg'])
    st.dataframe(career_stats)

    st.header("Yearly Stats")
    yearly_stats = player_data.copy()
    yearly_stats['Name'] = yearly_stats['IDfg'].map(id_to_name)
    st.dataframe(yearly_stats.set_index(['Name', 'IDfg', 'year']))

    st.header("Stat Explorer")
    numeric_columns = player_data.select_dtypes(include=['int64', 'float64']).columns
    stat_options = [col for col in numeric_columns if col not in ['year', 'IDfg']]
    
    # Determine default stat based on player type
    if 'ERA' in stat_options:
        default_stat = 'ERA'
    elif 'H' in stat_options:
        default_stat = 'H'
    else:
        default_stat = stat_options[0]  # Fallback to first available stat

    col1, col2 = st.columns(2)
    with col1:
        selected_stat = st.selectbox("Choose a stat to visualize:", stat_options, index=stat_options.index(default_stat))
    with col2:
        chart_type = st.radio("Select chart type:", ["Line"])

    if chart_type == "Line":
        fig = px.line(player_data, x='year', y=selected_stat, color='IDfg', 
                      title=f"{selected_stat} Over Time")
    
    for trace in fig.data:
        idfg = int(trace.name)
        player_subset = player_data[player_data['IDfg'] == idfg]
        if not player_subset.empty:
            player_info = player_subset.iloc[-1]
            trace.name = player_info['Name']
            trace.line.color = get_team_color(player_info['Team'], team_colors)
        else:
            trace.name = f"Unknown Player (ID: {idfg})"
            trace.line.color = 'grey'
    
    st.plotly_chart(fig)

    cumulative_data = []
    for idfg, group in player_data.groupby('IDfg'):
        group = group.sort_values('year')
        group['career_year'] = range(1, len(group) + 1)
        group[f'Cumulative_{selected_stat}'] = group[selected_stat].cumsum()
        cumulative_data.append(group)
    
    cumulative_df = pd.concat(cumulative_data)
    
    fig_cumulative = px.line(cumulative_df, x='career_year', y=f'Cumulative_{selected_stat}', 
                             color='IDfg',
                             title=f"Cumulative {selected_stat} Over Career Years",
                             labels={'career_year': 'Career Year'})
    
    for trace in fig_cumulative.data:
        idfg = int(trace.name)
        player_subset = cumulative_df[cumulative_df['IDfg'] == idfg]
        if not player_subset.empty:
            player_info = player_subset.iloc[-1]
            trace.name = player_info['Name']
            trace.line.color = get_team_color(player_info['Team'], team_colors)
        else:
            trace.name = f"Unknown Player (ID: {idfg})"
            trace.line.color = 'grey'
    
    st.plotly_chart(fig_cumulative)
def create_hover_text(row, stat, hover_data):
    hover_text = f"{row['year']}<br>{row['Team']}<br>{row['Name']}, {row[stat]}<br>"
    for key in hover_data:
        if key not in ['year', 'Team', 'Name', stat]:
            hover_text += f"{key}: {row[key]}<br>"
    return hover_text.rstrip('<br>')

def plot_league_wide_stat(df, stat, year_range, stat_min_max, hover_data, data_type):
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    if stat_min_max == 'min':
        selected_stat = df.loc[df.groupby('year')[stat].idxmin()]
    else:
        selected_stat = df.loc[df.groupby('year')[stat].idxmax()]
    
    team_colors = load_team_colors()
    selected_stat['color'] = selected_stat['Team'].map(lambda x: get_team_color(x, team_colors))
    selected_stat['hover_text'] = selected_stat.apply(lambda row: create_hover_text(row, stat, hover_data), axis=1)

    fig = go.Figure()
    
    for team in sorted(selected_stat['Team'].unique()):
        team_data = selected_stat[selected_stat['Team'] == team]
        fig.add_trace(go.Bar(
            x=team_data['year'],
            y=team_data[stat],
            name=team,
            marker_color=get_team_color(team, team_colors),
            hoverinfo='text',
            hovertext=team_data['hover_text']
        ))

    fig.update_layout(
        title=f"{stat_min_max.capitalize()} {stat} per Year",
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
        )
    )


    st.plotly_chart(fig)

def league_wide_stats_view(data_type):
    st.subheader("League-wide Stat Histogram")
    
    data_df = load_and_filter_data(data_type)  # Load all data
    
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    available_stats = list(data_df.select_dtypes(include=[int, float]).columns)
    stat = st.selectbox("Select the stat you want to plot", available_stats)
    
    stat_min_max = st.radio("Plot max or min of the stat?", ('max', 'min'))
    
    if data_type == "Pitcher":
        hover_data = ['G', 'IP', 'ERA', 'WHIP', 'FIP', 'WAR']
    else:  # hitter
        hover_data = ['AVG', 'OBP', 'OPS', 'HR', 'wRC+', 'SB', 'WAR']
    
    if st.button("Generate Histogram"):
        plot_league_wide_stat(data_df, stat, year_range, stat_min_max, hover_data, data_type)

def process_data_for_race(df, stat, start_year, end_year):
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if df[stat].isnull().all():
        raise ValueError(f"No {stat} data available for the selected range.")
    
    player_stats = df.groupby(['IDfg', 'year'])[stat].sum().unstack(fill_value=0)
    player_stats_cumsum = player_stats.cumsum(axis=1)
    
    id_to_name = df.sort_values('year').groupby('IDfg').last()[['Name', 'Team']]
    
    data_for_animation = []
    for year in range(start_year, end_year + 1):
        year_data = player_stats_cumsum[year].sort_values(ascending=False).head(10)
        for rank, (idfg, value) in enumerate(year_data.items(), 1):
            name = id_to_name.loc[idfg, 'Name']
            team = id_to_name.loc[idfg, 'Team']
            data_for_animation.append({
                'Year': year,
                'IDfg': idfg,
                'Name': name,
                'Value': value,
                'Rank': rank,
                'Team': team
            })
    
    return pd.DataFrame(data_for_animation)

def create_race_plot(df, stat, start_year, end_year):
    team_colors = load_team_colors()
    color_map = {idfg: get_team_color(df[df['IDfg'] == idfg]['Team'].iloc[0], team_colors) for idfg in df['IDfg'].unique()}

    fig = go.Figure(
        data=[
            go.Bar(
                x=df[df['Year'] == start_year]['Value'],
                y=df[df['Year'] == start_year]['Name'],
                orientation='h',
                text=df[df['Year'] == start_year]['Value'],
                texttemplate='%{text:.0f}',
                textposition='outside',
                marker=dict(color=[color_map[idfg] for idfg in df[df['Year'] == start_year]['IDfg']])
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f"Top 10 {stat} Totals Over Time",
                font=dict(size=24)
            ),
            xaxis=dict(range=[0, df['Value'].max() * 1.1], autorange=False, title=stat),
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
            x=df[df['Year'] == year]['Value'],
            y=df[df['Year'] == year]['Name'],
            orientation='h',
            text=df[df['Year'] == year]['Value'],
            marker=dict(color=[color_map[idfg] for idfg in df[df['Year'] == year]['IDfg']])
        )],
        layout=go.Layout(
            yaxis=dict(
                categoryorder='array',
                categoryarray=df[df['Year'] == year]['Name'][::-1],
                tickfont=dict(weight='bold')
            )
        ),
        name=str(year)
    ) for year in range(start_year, end_year + 1)]

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
                for year in range(start_year, end_year + 1)
            ]
        }]
    )

    st.plotly_chart(fig)

def race_chart_view(data_type):
    st.subheader("Career Stat Race Chart")
    
    data_df = load_and_filter_data(data_type)  # Load all data
    
    min_year, max_year = int(data_df['year'].min()), int(data_df['year'].max())
    start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year, value=min_year)
    end_year = st.number_input("End Year", min_value=min_year, max_value=max_year, value=max_year)
    
    if start_year >= end_year:
        st.error("Start year must be less than end year.")
        return
    
    available_stats = list(data_df.select_dtypes(include=[int, float]).columns)
    stat = st.selectbox("Select the stat for the race chart", available_stats)
    
    if st.button("Generate Race Chart"):
        try:
            processed_data = process_data_for_race(data_df, stat, start_year, end_year)
            create_race_plot(processed_data, stat, start_year, end_year)
        except ValueError as e:
            st.error(str(e))

def calculate_similarity_scores(player_data, target_player, stats_to_compare):
    # Filter players of the same type (hitter or pitcher)
    player_data = player_data[player_data['player_type'] == target_player['player_type']]
    
    # Prepare the data
    players_stats = player_data.groupby('IDfg')[stats_to_compare].mean().reset_index()
    
    # Remove players with missing data for any of the selected stats
    players_stats = players_stats.dropna(subset=stats_to_compare)
    
    # If target player is not in the filtered dataset, return empty DataFrame
    if target_player['IDfg'] not in players_stats['IDfg'].values:
        return pd.DataFrame(columns=['IDfg', 'Similarity'])
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_stats = scaler.fit_transform(players_stats[stats_to_compare])
    
    # Calculate distances
    target_player_stats = normalized_stats[players_stats['IDfg'] == target_player['IDfg']]
    distances = euclidean_distances(target_player_stats, normalized_stats)[0]
    
    # Create similarity scores (inverse of distance)
    max_distance = np.max(distances)
    similarity_scores = 1 - (distances / max_distance)
    
    # Create a dataframe with results
    results = pd.DataFrame({
        'IDfg': players_stats['IDfg'],
        'Similarity': similarity_scores
    })
    
    # Sort by similarity and exclude the target player
    results = results[results['IDfg'] != target_player['IDfg']].sort_values('Similarity', ascending=False)
    
    return results

def player_similarity_view(data_type):
    st.subheader("Player Similarity Scores")

    player_type = st.radio("Would you like to find similar hitters or pitchers?", ("Hitters", "Pitchers"))
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    all_data = load_and_filter_data(data_type)
    players = all_data[['IDfg', 'Name', 'player_type']].drop_duplicates()
    
    target_player_name = st.selectbox(f"Select a {player_type.lower()[:-1]}:", players['Name'].unique())
    num_similar_players = st.slider("Number of similar players to find:", 1, 20, 5)
    
    if data_type == "Pitcher":
        default_stats = ['ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP']
    else:
        default_stats = ['AVG', 'OBP', 'SLG', 'wRC+', 'ISO', 'BB%', 'K%']
    
    available_stats = [col for col in all_data.columns if col not in ['IDfg', 'Name', 'Team', 'year', 'player_type']]
    stats_to_compare = st.multiselect("Select stats to compare:", available_stats, default=default_stats)
    
    if st.button("Find Similar Players"):
        target_player = players[players['Name'] == target_player_name].iloc[0]
        similarity_scores = calculate_similarity_scores(all_data, target_player, stats_to_compare)
        
        if similarity_scores.empty:
            st.warning(f"No similar players found for {target_player_name} using the selected stats. This may be due to missing data for the selected player or stats. Try selecting different stats.")
        else:
            st.subheader(f"Players most similar to {target_player_name}")
            for _, player in similarity_scores.head(num_similar_players).iterrows():
                similar_player = players[players['IDfg'] == player['IDfg']].iloc[0]
                st.write(f"{similar_player['Name']} (Similarity: {player['Similarity']:.2f})")
            
            # WAR Comparison Plot
            war_data = []
            for _, player in similarity_scores.head(num_similar_players).iterrows():
                player_data = all_data[all_data['IDfg'] == player['IDfg']]
                war = player_data['WAR'].mean() if 'WAR' in player_data.columns else 0
                war_data.append({
                    'Name': players[players['IDfg'] == player['IDfg']].iloc[0]['Name'],
                    'WAR': war,
                    'Similarity': player['Similarity']
                })
            
            # Add target player to the WAR data
            target_player_data = all_data[all_data['IDfg'] == target_player['IDfg']]
            target_war = target_player_data['WAR'].mean() if 'WAR' in target_player_data.columns else 0
            war_data.append({
                'Name': target_player_name,
                'WAR': target_war,
                'Similarity': 1.0  # Perfect similarity with itself
            })
            
            war_df = pd.DataFrame(war_data)
            war_df = war_df.sort_values('Similarity', ascending=False)
            
            fig = px.bar(war_df, x='Name', y='WAR', 
                         title=f"WAR Comparison: {target_player_name} vs Similar Players",
                         labels={'WAR': 'Average WAR', 'Name': 'Player'},
                         color='Similarity', color_continuous_scale='viridis')
            
            fig.update_layout(xaxis_tickangle=-45, xaxis_title="")
            st.plotly_chart(fig)

    st.info(f"Note: This similarity comparison is for {player_type.lower()} only. To find similar {'pitchers' if player_type == 'Hitters' else 'hitters'}, please start a new similarity search.")

def main():
    st.title("Brock's Baseball Stats Explorer")
    
    st.header("It's Time For Dodgers Baseball! (or your team who sucks)")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the mode",
                                ["Individual Player", "Compare Players", "League-wide Stats", "Career Stat Race", "Player Similarity"])
    
    # Reset session state when changing modes
    if 'current_mode' not in st.session_state or st.session_state.current_mode != app_mode:
        st.session_state.player_data = None
        st.session_state.current_mode = app_mode
    
    data_type = st.sidebar.radio("Select player type:", ("Hitter", "Pitcher"))
    
    if app_mode == "Individual Player":
        individual_player_view(data_type)
    elif app_mode == "Compare Players":
        compare_players_view(data_type)
    elif app_mode == "League-wide Stats":
        league_wide_stats_view(data_type)
    elif app_mode == "Career Stat Race":
        race_chart_view(data_type)
    else:  # Player Similarity
        player_similarity_view(data_type)

if __name__ == "__main__":
    main()