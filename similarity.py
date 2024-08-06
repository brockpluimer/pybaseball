import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict, Union
from load_data import load_and_prepare_data

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

    data_df, player_years = load_and_prepare_data(data_type)
    
    # Set Clayton Kershaw as the default pitcher and Shohei Ohtani as the default hitter
    default_player = "Clayton Kershaw" if data_type == "Pitcher" else "Shohei Ohtani"
    
    # Check if the default player is in the dataset, if not use the first player in the list
    if default_player not in player_years['Name'].unique():
        default_player = player_years['Name'].iloc[0]
    
    # Create a dictionary mapping player labels to their IDfg
    player_label_to_id = dict(zip(player_years['Label'], player_years['IDfg']))
    
    target_player_label = st.selectbox(
        f"Select a {player_type.lower()[:-1]}:",
        player_years['Label'],
        index=player_years['Label'].tolist().index(player_years[player_years['Name'] == default_player]['Label'].iloc[0]),
        key="player_name_selectbox"
    )
    
    target_player_id = player_label_to_id[target_player_label]
    target_player_name = player_years[player_years['IDfg'] == target_player_id]['Name'].iloc[0]
    
    if mode == "Season":
        seasons = sorted(data_df[data_df['IDfg'] == target_player_id]['year'].unique(), reverse=True)
        target_year = st.selectbox(
            "Select season:",
            seasons,
            index=0,
            key="season_selectbox"
        )
        target_player = data_df[(data_df['IDfg'] == target_player_id) & (data_df['year'] == target_year)].iloc[0]
    else:
        target_player = player_years[player_years['IDfg'] == target_player_id].iloc[0]

    num_similar_players = st.slider("Number of similar players to find:", 1, 20, 5, key="num_similar_players_slider")
    
    if data_type == "Pitcher":
        default_stats = ['WAR', 'ERA', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'FIP']
    else:
        default_stats = ['WAR', 'AVG', 'OBP', 'SLG', 'wRC+', 'ISO', 'BB%', 'K%']
    
    available_stats = [col for col in data_df.columns if col not in ['IDfg', 'Name', 'Team', 'year', 'player_type']]
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
        similarity_scores = calculate_similarity_scores(data_df, target_player, stats_to_compare, mode.lower(), scaling_factor)
        
        if similarity_scores.empty:
            st.warning(f"No similar players found for {target_player_label} using the selected stats. This may be due to missing data for the selected player or stats. Try selecting different stats.")
        else:
            st.subheader(f"Players most similar to {target_player_label}")
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
                title=f"Top {num_similar_players} Similar Players to {target_player_label} ({mode} Comparison)"
            )

            # Add target player as a different marker
            target_data = pd.DataFrame([target_player])
            if y_stat not in target_data.columns:
                # Use the mean of the stat for the target player if not available
                target_data[y_stat] = data_df[data_df['IDfg'] == target_player['IDfg']][y_stat].mean()
            
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
