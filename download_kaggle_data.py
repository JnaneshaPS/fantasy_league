import os
import json
import pandas as pd
import numpy as np
import shutil
import requests
import zipfile
import io

def download_ipl_data():
    """Download IPL datasets from Kaggle or alternative sources"""
    print("Attempting to download IPL data...")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/downloads', exist_ok=True)
    
    # Try to download from Cricsheet (doesn't require authentication)
    try:
        print("Downloading IPL data from Cricsheet (no auth required)...")
        url = "https://cricsheet.org/downloads/ipl_json.zip"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("Successfully downloaded data from Cricsheet")
            
            # Extract the zip file
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall("data/downloads/cricsheet")
            
            # Process the JSON files to CSV
            process_cricsheet_data()
            return True
    except Exception as e:
        print(f"Error downloading from Cricsheet: {e}")
    
    # Fall back to sample data
    print("Creating sample data instead")
    create_sample_data()
    return False

def process_cricsheet_data():
    """Process Cricsheet JSON data into CSV format"""
    import json
    import os
    import glob
    
    print("Processing Cricsheet JSON data...")
    
    # Get list of JSON files
    json_files = glob.glob("data/downloads/cricsheet/*.json")
    
    if not json_files:
        print("No JSON files found in cricsheet data")
        return
        
    print(f"Found {len(json_files)} JSON match files")
    
    # Create data structures for matches and players
    matches = []
    players_data = {}
    teams_data = {}
    
    # Process each match
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                match_data = json.load(f)
            
            # Extract match info
            match_info = match_data.get('info', {})
            match_id = os.path.basename(json_file).split('.')[0]
            
            # Basic match details
            match = {
                'match_id': match_id,
                'date': match_info.get('dates', ['Unknown'])[0],
                'teams': match_info.get('teams', []),
                'venue': match_info.get('venue', 'Unknown'),
                'city': match_info.get('city', 'Unknown'),
                'toss_winner': match_info.get('toss', {}).get('winner', 'Unknown'),
                'toss_decision': match_info.get('toss', {}).get('decision', 'Unknown'),
                'winner': match_info.get('outcome', {}).get('winner', 'Unknown')
            }
            
            # Add to matches list
            matches.append(match)
            
            # Process player data
            player_lists = match_info.get('players', {})
            for team, players in player_lists.items():
                # Update team data
                if team not in teams_data:
                    teams_data[team] = {'name': team, 'matches': 0}
                teams_data[team]['matches'] += 1
                
                # Update player data
                for player in players:
                    if player not in players_data:
                        players_data[player] = {
                            'player_name': player,
                            'team': team,
                            'matches': 0,
                            'innings': 0,
                            'runs': 0,
                            'batting_average': 0,
                            'strike_rate': 0,
                            'role': 'Unknown'  # We'll try to infer this later
                        }
                    players_data[player]['matches'] += 1
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Save to CSV
    pd.DataFrame(matches).to_csv('data/raw/match_stats.csv', index=False)
    pd.DataFrame(list(players_data.values())).to_csv('data/raw/batting_stats.csv', index=False)
    pd.DataFrame(list(teams_data.values())).to_csv('data/raw/team_stats.csv', index=False)
    
    print(f"Processed {len(matches)} matches and {len(players_data)} players")
    
def create_sample_data():
    """Create sample IPL data if downloads fail"""
    print("Creating sample IPL data...")
    
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Sample teams
    teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", 
             "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings", 
             "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"]
    
    # Sample players with positions
    player_roles = {
        'WK': 10,    # 10 wicket-keepers
        'BAT': 40,   # 40 batsmen 
        'BOWL': 30,  # 30 bowlers
        'AR': 20     # 20 all-rounders
    }
    
    # Generate player data
    players = []
    player_id = 1
    
    for role, count in player_roles.items():
        for i in range(count):
            team = teams[player_id % len(teams)]
            
            # Different stat ranges based on role
            if role == 'BAT':
                avg_runs = np.random.randint(30, 60)
                sr = np.random.randint(120, 180)
                economy = 0
            elif role == 'BOWL':
                avg_runs = np.random.randint(10, 25)
                sr = np.random.randint(100, 140)
                economy = np.random.uniform(6.5, 9.5)
            elif role == 'AR':
                avg_runs = np.random.randint(20, 40)
                sr = np.random.randint(130, 160)
                economy = np.random.uniform(7.5, 10.5)
            else:  # WK
                avg_runs = np.random.randint(25, 45)
                sr = np.random.randint(125, 170)
                economy = 0
            
            player = {
                'player_id': player_id,
                'player_name': f"{team.split()[0]} Player {player_id}",
                'team': team,
                'role': role,
                'matches': np.random.randint(20, 100),
                'innings': np.random.randint(15, 90),
                'runs': np.random.randint(200, 3000),
                'batting_average': avg_runs,
                'strike_rate': sr,
                'economy_rate': economy,
                'wickets': np.random.randint(0, 80) if role in ['BOWL', 'AR'] else 0
            }
            players.append(player)
            player_id += 1
    
    # Generate match data
    matches = []
    for i in range(1, 60):
        home_idx = i % len(teams)
        away_idx = (i + 1) % len(teams)
        match = {
            'match_id': i,
            'date': f"2023-04-{(i%30)+1:02d}",
            'home_team': teams[home_idx],
            'away_team': teams[away_idx],
            'venue': f"{teams[home_idx].split()[0]} Stadium",
            'winner': teams[home_idx] if i % 3 != 0 else teams[away_idx]
        }
        matches.append(match)
    
    # Save the data
    pd.DataFrame(players).to_csv('data/raw/batting_stats.csv', index=False)
    pd.DataFrame(matches).to_csv('data/raw/match_stats.csv', index=False)
    
    print(f"Created sample data with {len(players)} players and {len(matches)} matches")
    return True

if __name__ == "__main__":
    download_ipl_data()