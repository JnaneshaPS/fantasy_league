import os
import pandas as pd
from data_integration import preprocess_performance_data, train_prediction_models, create_player_mappings, download_kaggle_dataset

def initialize_data():
    print("Initializing fantasy cricket prediction system with real data...")
    
    # Check if we already have processed data
    if os.path.exists('data/processed/batsmen_stats.csv') and os.path.exists('data/processed/bowlers_stats.csv'):
        print("Found already processed cricket data, using that...")
        batsmen_df = pd.read_csv('data/processed/batsmen_stats.csv')
        bowlers_df = pd.read_csv('data/processed/bowlers_stats.csv')
        
        # Train models
        print("Training prediction models...")
        models = train_prediction_models(batsmen_df, bowlers_df)
        
        # Create player mappings
        print("Creating player name mappings...")
        mappings = create_player_mappings(batsmen_df, bowlers_df)
        
        print("Data initialization complete! System is ready for accurate Dream11 predictions.")
        return True
    
    # Try to download and process from Kaggle
    print("Dataset URL: https://www.kaggle.com/datasets/akarshsinghh/cricket-player-performance-prediction")
    success = download_kaggle_dataset()
    
    if success:
        # Process the data
        print("Processing cricket performance data...")
        batsmen_df, bowlers_df = preprocess_performance_data()
        
        if batsmen_df is not None and bowlers_df is not None:
            # Save processed data
            os.makedirs('data/processed', exist_ok=True)
            batsmen_df.to_csv('data/processed/batsmen_stats.csv', index=False)
            bowlers_df.to_csv('data/processed/bowlers_stats.csv', index=False)
            
            # Train ML models
            print("Training prediction models...")
            models = train_prediction_models(batsmen_df, bowlers_df)
            
            # Create player name mappings
            print("Creating player name mappings...")
            mappings = create_player_mappings(batsmen_df, bowlers_df)
            
            print("Data initialization complete! System is ready for accurate Dream11 predictions.")
            return True
    
    print("Data initialization failed. Using synthetic data for predictions.")
    return False

if __name__ == "__main__":
    initialize_data()