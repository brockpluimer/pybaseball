import os
import pandas as pd
import plotly.graph_objects as go

# Load team colors from the text file
team_colors = {}
with open('team_colors.txt', 'r') as file:
    for line in file:
        team, color = line.strip().split(': ')
        team_colors[team.upper()] = color

def load_data(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            year = filename.split('_')[-1].split('.')[0]
            file_path = os.path.join(data_dir, filename)
            data = pd.read_csv(file_path)
            data['year'] = int(year)
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

def create_hover_text(row, stat, hover_data):
    hover_text = f"{row['year']}<br>{row['Team']}<br>{row['Name']}, {row[stat]}<br>"
    for key in hover_data:
        if key not in ['year', 'Team', 'Name', stat]:
            hover_text += f"{key}: {row[key]}<br>"
    return hover_text.rstrip('<br>')

def plot_data(df, stat, year_range, stat_min_max, hover_data, data_type):
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    if stat_min_max == 'min':
        selected_stat = df.loc[df.groupby('year')[stat].idxmin()]
    else:
        selected_stat = df.loc[df.groupby('year')[stat].idxmax()]
    
    selected_stat['color'] = selected_stat['Team'].map(team_colors).fillna('grey')
    selected_stat['hover_text'] = selected_stat.apply(lambda row: create_hover_text(row, stat, hover_data), axis=1)

    fig = go.Figure()
    
    for team in sorted(selected_stat['Team'].unique()):
        team_data = selected_stat[selected_stat['Team'] == team]
        fig.add_trace(go.Bar(
            x=team_data['year'],
            y=team_data[stat],
            name=team,
            marker_color=team_colors.get(team, 'grey'),
            hoverinfo='text',
            hovertext=team_data['hover_text']
        ))

    fig.update_layout(
        title=f"{stat_min_max.capitalize()} {stat} per Year",
        xaxis_title="Year",
        yaxis_title=stat,
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=14),
        xaxis=dict(type='category', categoryorder='category ascending'),
        legend_title_text='Team',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    fig.show()

def main():
    data_type = input("Do you want to analyze pitcher or hitter data? (pitcher/hitter): ").strip().lower()
    if data_type not in ['pitcher', 'hitter']:
        print("Invalid input. Please enter 'pitcher' or 'hitter'.")
        return
    
    data_dir = 'pitcher_data' if data_type == 'pitcher' else 'hitter_data'
    data_df = load_data(data_dir)
    
    min_year, max_year = data_df['year'].min(), data_df['year'].max()
    print(f"Available year range: {min_year} - {max_year}")
    
    start_year = int(input(f"Enter the start year (between {min_year} and {max_year}): ").strip())
    end_year = int(input(f"Enter the end year (between {min_year} and {max_year}): ").strip())
    
    if start_year < min_year or end_year > max_year or start_year > end_year:
        print("Invalid year range.")
        return
    
    available_stats = list(data_df.select_dtypes(include=[int, float]).columns)
    print(f"Available stats: {', '.join(available_stats)}")
    
    stat = input("Enter the stat you want to plot: ").strip()
    if stat not in available_stats:
        print("Invalid stat.")
        return
    
    stat_min_max = input("Do you want to plot the max or the min of the stat? (max/min): ").strip().lower()
    if stat_min_max not in ['max', 'min']:
        print("Invalid input. Please enter 'max' or 'min'.")
        return

    if data_type == 'pitcher':
        hover_data = ['G', 'IP', 'ERA', 'WHIP', 'FIP', 'WAR']
    else:  # hitter
        hover_data = ['AVG', 'OBP', 'OPS', 'HR', 'wRC+', 'SB', 'WAR']

    plot_data(data_df, stat, (start_year, end_year), stat_min_max, hover_data, data_type)

if __name__ == "__main__":
    main()