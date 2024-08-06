import streamlit as st
from load_data import load_and_filter_data

def milestone_tracker():
    st.subheader("Milestone Tracker")

    st.markdown("""
    This tool allows you to search for players who have achieved specific statistical milestones, 
    either in a single season or over their entire career. Here's how to use it:

    1. Choose whether you want to track milestones for hitters or pitchers.
    2. Select whether you're looking for a single-season or career milestone.
    3. Choose the statistic you're interested in.
    4. Set the milestone value (e.g., 3000 for 3000 career hits).
    5. Optionally, set minimum playing time requirements to filter out small sample sizes.
    6. The tool will then display all players who have achieved this milestone.

    This can be used to find rare achievements, track historical performances, or see how close 
    current players are to reaching significant milestones.
    """)

    player_type = st.radio("Select player type:", ("Hitters", "Pitchers"), key="milestone_player_type")
    data_type = "Hitter" if player_type == "Hitters" else "Pitcher"

    milestone_type = st.radio("Select milestone type:", ("Single Season", "Career"), key="milestone_type")

    data_df = load_and_filter_data(data_type)

    if data_type == "Hitter":
        default_stats = ['G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'WAR']
    else:  # Pitcher
        default_stats = ['W', 'L', 'ERA', 'G', 'GS', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'WHIP', 'K/9', 'BB/9', 'FIP', 'WAR']

    stat = st.selectbox("Select the stat for the milestone:", default_stats, key="milestone_stat")

    milestone_value = st.number_input("Enter the milestone value:", min_value=0.0, step=0.1, key="milestone_value")

    use_min_filter = st.checkbox("Set minimum playing time filter?", key="milestone_min_filter")
    min_pa = min_ip = None
    if use_min_filter:
        if data_type == "Hitter":
            min_pa = st.number_input("Minimum PA:", min_value=1, value=300, key="milestone_min_pa")
        else:  # Pitcher
            min_ip = st.number_input("Minimum IP:", min_value=1, value=50, key="milestone_min_ip")

    if st.button("Find Players Who Reached This Milestone"):
        if milestone_type == "Single Season":
            if use_min_filter:
                if data_type == "Hitter":
                    filtered_df = data_df[data_df['PA'] >= min_pa]
                else:  # Pitcher
                    filtered_df = data_df[data_df['IP'] >= min_ip]
            else:
                filtered_df = data_df

            milestone_players = filtered_df[filtered_df[stat] >= milestone_value].sort_values(stat, ascending=False)
            
            if not milestone_players.empty:
                st.success(f"Found {len(milestone_players)} player seasons that reached this milestone!")
                
                # Create a list of columns to display, ensuring no duplicates
                display_columns = ['Name', 'year', 'Team'] + [col for col in default_stats if col != stat]
                if stat not in display_columns:
                    display_columns.insert(3, stat)  # Insert the stat after 'Team' if it's not already included
                
                st.dataframe(milestone_players[display_columns])
            else:
                st.warning("No players found who reached this milestone in a single season.")

        else:  # Career milestone
            career_stats = data_df.groupby('Name').agg({
                stat: 'sum' if stat not in ['AVG', 'OBP', 'SLG', 'OPS', 'ERA', 'WHIP', 'K/9', 'BB/9', 'FIP'] else 'mean',
                'year': ['min', 'max']
            }).reset_index()
            career_stats.columns = ['Name', stat, 'First Year', 'Last Year']
            career_stats['Years Played'] = career_stats['Last Year'] - career_stats['First Year'] + 1

            if use_min_filter:
                career_totals = data_df.groupby('Name').agg({
                    'PA' if data_type == "Hitter" else 'IP': 'sum'
                }).reset_index()
                career_stats = career_stats.merge(career_totals, on='Name')
                if data_type == "Hitter":
                    career_stats = career_stats[career_stats['PA'] >= min_pa * career_stats['Years Played']]
                else:  # Pitcher
                    career_stats = career_stats[career_stats['IP'] >= min_ip * career_stats['Years Played']]

            milestone_players = career_stats[career_stats[stat] >= milestone_value].sort_values(stat, ascending=False)

            if not milestone_players.empty:
                st.success(f"Found {len(milestone_players)} players who reached this career milestone!")
                st.dataframe(milestone_players)
            else:
                st.warning("No players found who reached this career milestone.")