import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import glob

def download_kaggle_dataset():
    """Download the cricket player performance dataset from Kaggle"""
    try:
        import kaggle
        # Create data directory if it doesn't exist
        os.makedirs('data/kaggle', exist_ok=True)
        
        # Dataset info
        dataset_name = 'akarshsinghh/cricket-player-performance-prediction'
        print(f"Dataset URL: https://www.kaggle.com/datasets/{dataset_name}")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path='data/kaggle',
            unzip=True
        )
        print("Successfully downloaded Kaggle cricket performance dataset")
        
        # List all downloaded CSV files
        csv_files = glob.glob('data/kaggle/*.csv')
        print(f"Downloaded files: {csv_files}")
        
        if not csv_files:
            # Check if files might be in a subdirectory
            subdirs = [d for d in os.listdir('data/kaggle') if os.path.isdir(os.path.join('data/kaggle', d))]
            for subdir in subdirs:
                subdir_path = os.path.join('data/kaggle', subdir)
                csv_files = glob.glob(f'{subdir_path}/*.csv')
                if csv_files:
                    print(f"Found CSV files in subdirectory: {subdir}")
                    # Move files to main directory
                    for file in csv_files:
                        filename = os.path.basename(file)
                        os.rename(file, os.path.join('data/kaggle', filename))
                    print(f"Moved files to data/kaggle directory")
                    break
        
        if csv_files:
            return True
        else:
            print("No CSV files found in the downloaded dataset")
            return False
            
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please ensure you have kaggle API credentials in ~/.kaggle/kaggle.json")
        return False

def preprocess_performance_data():
    """Preprocess the Kaggle cricket performance data"""
    # Look for any CSV files in the data/kaggle directory
    data_path = 'data/kaggle'
    csv_files = glob.glob(f'{data_path}/*.csv')
    
    if not csv_files:
        print("Kaggle data files not found. Please download first.")
        return None, None
    
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # First, let's examine what columns we actually have in each file
    for file in csv_files:
        df = pd.read_csv(file, nrows=1)
        print(f"File: {os.path.basename(file)}")
        print(f"Columns: {list(df.columns)}")
    
    # Try to determine which files are which based on column contents
    batsmen_df = None
    bowlers_df = None
    matches_df = None
    
    for file in csv_files:
        # Read the file
        df = pd.read_csv(file)
        filename = os.path.basename(file).lower()
        
        if 'bat' in filename:
            print(f"Using {filename} as batsmen data")
            batsmen_df = df
        elif 'ball' in filename:
            print(f"Using {filename} as bowlers data")
            bowlers_df = df
        elif 'match' in filename:
            print(f"Using {filename} as match data")
            matches_df = df
    
    # Process data with adapted functions that match the actual dataset structure
    if batsmen_df is not None:
        processed_batsmen = process_batsmen_data_adapted(batsmen_df, matches_df)
    else:
        processed_batsmen = None
    
    if bowlers_df is not None:
        processed_bowlers = process_bowlers_data_adapted(bowlers_df, matches_df)
    else:
        processed_bowlers = None
    
    return processed_batsmen, processed_bowlers

def process_batsmen_data_adapted(df, matches_df=None):
    """Process batsmen data with columns adapted to the actual dataset"""
    # Print sample data to understand structure
    print("Batsmen data sample:")
    print(df.head(1))
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df = df.fillna(0)
    
    # Fix 1: Adapt column names to what's actually in the dataset
    run_col = None
    if 'runs_x' in df.columns:
        run_col = 'runs_x'
    elif 'runs' in df.columns:
        run_col = 'runs'
        
    if not run_col:
        print("ERROR: Could not find a column for runs")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Fix 2: Use name_x as the batsman column
    batsman_col = 'name_x'
    if batsman_col not in df.columns:
        print("ERROR: Could not find batsman name column")
        return None
    
    # Group by batsman to get aggregate stats
    batsmen_stats = df.groupby(batsman_col).agg({
        run_col: 'sum',
    }).reset_index()
    
    # Rename columns for consistency
    batsmen_stats = batsmen_stats.rename(columns={
        batsman_col: 'Batsman',
        run_col: 'Runs'
    })
    
    # Calculate fantasy points based on available data
    batsmen_stats['fantasy_points'] = batsmen_stats['Runs'] * 1  # 1 point per run
    
    # Add fours and sixes if available
    if 'fours' in df.columns:
        fours_stats = df.groupby(batsman_col)['fours'].sum().reset_index()
        fours_stats.columns = ['Batsman', '4s']
        batsmen_stats = batsmen_stats.merge(fours_stats, on='Batsman', how='left')
        batsmen_stats['4s'] = batsmen_stats['4s'].fillna(0)
        batsmen_stats['fantasy_points'] += batsmen_stats['4s'] * 1  # 1 point per boundary
    else:
        batsmen_stats['4s'] = 0
    
    if 'sixes' in df.columns:
        sixes_stats = df.groupby(batsman_col)['sixes'].sum().reset_index()
        sixes_stats.columns = ['Batsman', '6s']
        batsmen_stats = batsmen_stats.merge(sixes_stats, on='Batsman', how='left')
        batsmen_stats['6s'] = batsmen_stats['6s'].fillna(0)
        batsmen_stats['fantasy_points'] += batsmen_stats['6s'] * 2  # 2 points per six
    else:
        batsmen_stats['6s'] = 0
    
    # Add balls faced
    if 'balls' in df.columns:
        balls_stats = df.groupby(batsman_col)['balls'].sum().reset_index()
        balls_stats.columns = ['Batsman', 'Balls']
        batsmen_stats = batsmen_stats.merge(balls_stats, on='Batsman', how='left')
    else:
        # Estimate balls faced from runs (very rough estimate)
        batsmen_stats['Balls'] = batsmen_stats['Runs'] * 0.75
    
    # Calculate strike rate
    batsmen_stats['SR'] = (batsmen_stats['Runs'] / batsmen_stats['Balls']) * 100
    batsmen_stats['SR'] = batsmen_stats['SR'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Add form data (simplest version)
    batsmen_stats['recent_form'] = batsmen_stats['Runs'] 
    
    return batsmen_stats

def process_bowlers_data_adapted(df, matches_df=None):
    """Process bowlers data with columns adapted to the actual dataset"""
    # Print sample data to understand structure
    print("Bowlers data sample:")
    print(df.head(1))
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df = df.fillna(0)
    
    # Fix: use name_x as bowler column
    bowler_col = 'name_x'
    if bowler_col not in df.columns:
        print("ERROR: Could not find bowler column")
        return None
    
    # Use the right column names from the actual dataset
    wicket_col = 'wickets' if 'wickets' in df.columns else None
    runs_col = 'run_conceded' if 'run_conceded' in df.columns else 'runs' if 'runs' in df.columns else None
    overs_col = 'overs' if 'overs' in df.columns else None
    
    if not wicket_col or not runs_col or not overs_col:
        print(f"ERROR: Missing essential bowling columns. Found: wickets={wicket_col}, runs={runs_col}, overs={overs_col}")
        return None
    
    # Aggregate stats by bowler
    bowlers_stats = df.groupby(bowler_col).agg({
        wicket_col: 'sum',
        runs_col: 'sum',
        overs_col: 'sum'
    }).reset_index()
    
    # Rename columns for consistency
    bowlers_stats = bowlers_stats.rename(columns={
        bowler_col: 'Bowler',
        wicket_col: 'Wickets',
        runs_col: 'Runs',
        overs_col: 'Overs'
    })
    
    # Calculate economy rate
    bowlers_stats['Econ'] = bowlers_stats['Runs'] / bowlers_stats['Overs']
    bowlers_stats['Econ'] = bowlers_stats['Econ'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculate fantasy points based on Dream11 scoring system
    bowlers_stats['fantasy_points'] = (
        bowlers_stats['Wickets'] * 25 +    # 25 points per wicket
        (bowlers_stats['Econ'] < 6).astype(int) * 4  # Economy bonus
    )
    
    # Add maidens if we can detect them
    if 'maidens' in df.columns:
        maidens = df.groupby(bowler_col)['maidens'].sum().reset_index()
        maidens.columns = ['Bowler', 'Maidens']
        bowlers_stats = bowlers_stats.merge(maidens, on='Bowler', how='left')
        bowlers_stats['Maidens'] = bowlers_stats['Maidens'].fillna(0)
        bowlers_stats['fantasy_points'] += bowlers_stats['Maidens'] * 8  # 8 points per maiden
    else:
        bowlers_stats['Maidens'] = 0
    
    # Add recent form (simplified version)
    bowlers_stats['recent_form'] = bowlers_stats['Wickets']
    
    return bowlers_stats

def train_prediction_models(batsmen_df, bowlers_df):
    """Train ML models to predict player performance"""
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # Train batsmen model
    batsmen_features = ['Runs', 'Balls', '4s', '6s', 'SR', 'recent_form']
    batsmen_target = 'fantasy_points'
    
    # Handle missing values in features
    batsmen_X = batsmen_df[batsmen_features].fillna(0)
    batsmen_y = batsmen_df[batsmen_target]
    
    # Train a Random Forest model
    rf_batsmen = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_batsmen.fit(batsmen_X, batsmen_y)
    
    # Train a Gradient Boosting model (usually more accurate)
    gb_batsmen = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_batsmen.fit(batsmen_X, batsmen_y)
    
    # Save models
    joblib.dump(rf_batsmen, 'models/rf_batsmen.pkl')
    joblib.dump(gb_batsmen, 'models/gb_batsmen.pkl')
    
    # Train bowlers model
    bowlers_features = ['Overs', 'Runs', 'Wickets', 'Econ', 'recent_form']
    bowlers_target = 'fantasy_points'
    
    # Handle missing values in features
    bowlers_X = bowlers_df[bowlers_features].fillna(0)
    bowlers_y = bowlers_df[bowlers_target]
    
    # Train a Random Forest model
    rf_bowlers = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_bowlers.fit(bowlers_X, bowlers_y)
    
    # Train a Gradient Boosting model
    gb_bowlers = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_bowlers.fit(bowlers_X, bowlers_y)
    
    # Save models
    joblib.dump(rf_bowlers, 'models/rf_bowlers.pkl')
    joblib.dump(gb_bowlers, 'models/gb_bowlers.pkl')
    
    print("Successfully trained and saved prediction models")
    return {
        'batsmen': {'rf': rf_batsmen, 'gb': gb_batsmen},
        'bowlers': {'rf': rf_bowlers, 'gb': gb_bowlers}
    }

def create_player_mappings(batsmen_df, bowlers_df):
    """Create mappings between Kaggle dataset names and Dream11 names"""
    # Get unique player names from both datasets
    batsmen_names = set()
    if batsmen_df is not None and 'Batsman' in batsmen_df.columns:
        batsmen_names = set(batsmen_df['Batsman'].unique())
    
    bowler_names = set()
    if bowlers_df is not None and 'Bowler' in bowlers_df.columns:
        bowler_names = set(bowlers_df['Bowler'].unique())
    
    # Combine all player names
    all_players = batsmen_names.union(bowler_names)
    
    # Sort names for easier viewing
    all_players = sorted(list(all_players))
    
    print(f"Found {len(all_players)} unique players in the dataset")
    print("First 10 players:", all_players[:10])
    
    # Create a direct mapping (in a real system, you'd need to match with Dream11 names)
    name_mapping = {player: player for player in all_players}
    
    # Save mapping to file for future use
    os.makedirs('data/mappings', exist_ok=True)
    with open('data/mappings/player_name_mapping.txt', 'w') as f:
        for kaggle_name, dream11_name in name_mapping.items():
            f.write(f"{kaggle_name}|{dream11_name}\n")
    
    print(f"Created mapping file with {len(name_mapping)} player names")
    return name_mapping

if __name__ == "__main__":
    # Download dataset
    success = download_kaggle_dataset()
    
    if success:
        # Process data
        batsmen_df, bowlers_df = preprocess_performance_data()
        
        if batsmen_df is not None and bowlers_df is not None:
            # Train models
            models = train_prediction_models(batsmen_df, bowlers_df)
            
            # Create player mappings
            mappings = create_player_mappings(batsmen_df, bowlers_df)
            
            print("Data integration complete. Ready for enhanced predictions!")