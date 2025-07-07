import os
import pandas as pd
import numpy as np
import pickle
import time
import argparse
import random
import sys
from tabulate import tabulate

# Add processing directory to path
sys.path.insert(0, 'processing')

# Import the config
from processing.config import config

# Import utils functions
from processing.utils import (
    get_content_based_recommendations,
    get_text_based_recommendations,
    get_trending_recommendations,
    get_association_recommendations,
    get_als_recommendations,
    get_item_based_cf_recommendations,
    get_user_based_cf_recommendations
)

# Import other necessary modules
from processing.ALS_matrix_factorization import run_als, save_model
from processing.user_item_based_cf import compute_jaccard_similarity_matrix
from processing.evaluate_models import evaluate_als, evaluate_item_based_cf, evaluate_user_based_cf

def ensure_data_directories():
    """Create necessary data directories if they don't exist"""
    os.makedirs('processing/processedData', exist_ok=True)
    print("Checking data directories... OK")

def train_als_model(force=False):
    """Train the ALS model if not already trained"""
    print("\n=== Training ALS Model ===")
    from processing.ALS_matrix_factorization import load_data, save_model
    
    als_files = [
        'processing/processedData/U_matrix.pkl',
        'processing/processedData/V_matrix.pkl',
        'processing/processedData/user_biases.pkl',
        'processing/processedData/game_biases.pkl',
        'processing/processedData/user_idx.pkl',
        'processing/processedData/game_idx.pkl'
    ]
    
    # Skip if all files exist and force is False
    if not force and all(os.path.exists(f) for f in als_files):
        print("ALS model already trained. Use --force to retrain.")
        return
    
    print("Loading data...")
    user_item_matrix, user2game, game2user, user_game2rating, users, games, user_idx, game_idx, mu = load_data()
    
    print("Running ALS algorithm...")
    U, V, b, c = run_als(user2game, game2user, user_game2rating, users, games, user_idx, game_idx, mu)
    
    print("Saving ALS model...")
    save_model(U, V, b, c, user_idx, game_idx)
    
    print("ALS model training complete!")

def list_available_users():
    """List a sample of available users for testing"""
    print("\n=== Available Users for Testing ===")
    
    try:
        with open('processing/processedData/user2game_dict.pkl', 'rb') as f:
            user2game = pickle.load(f)
        
        # Get users with at least 5 games
        active_users = [user for user, games in user2game.items() if len(games) >= 5]
        
        # Select a random sample
        sample_size = min(10, len(active_users))
        sample_users = random.sample(active_users, sample_size)
        
        print(f"Found {len(active_users)} users with at least 5 games. Here's a sample of {sample_size}:")
        
        user_data = []
        for i, user in enumerate(sample_users):
            game_count = len(user2game[user])
            user_data.append([i+1, user, game_count])
        
        print(tabulate(user_data, headers=["#", "User ID", "Game Count"], tablefmt="pretty"))
        
        return sample_users
    except Exception as e:
        print(f"Error listing users: {e}")
        return []

def list_available_games():
    """List a sample of available games for testing from top popular games"""
    print("\n=== Available Games for Testing ===")
    
    try:
        # Get the game ratings for popularity
        if os.path.exists('processing/processedData/game_ratings_popularity.pkl'):
            game_ratings = pd.read_pickle('processing/processedData/game_ratings_popularity.pkl')
            
            # Sort by popularity (Bayesian rating or reviews count)
            if 'bayesian_rating' in game_ratings.columns:
                game_ratings = game_ratings.sort_values(by='bayesian_rating', ascending=False)
            elif 'reviews_count' in game_ratings.columns:
                game_ratings = game_ratings.sort_values(by='reviews_count', ascending=False)
                
            # Take top 1000 games or all if fewer
            top_games = game_ratings.head(1000)
            
            # Select a random sample from these popular games
            sample_size = min(10, len(top_games))
            game_sample = top_games.sample(sample_size)
            
            game_data = []
            for i, game in enumerate(game_sample.iterrows()):
                _, row = game
                game_data.append([i+1, row['id'], row['app_name']])
            
            print(f"Showing a random sample of {sample_size} popular games:")
            print(tabulate(game_data, headers=["#", "Game ID", "Game Name"], tablefmt="pretty"))
            
            return game_sample
        else:
            # Fallback to using games data directly if ratings not available
            print("Game ratings not found, using all games instead.")
            games = config.games
            sample_size = 10
            game_sample = games.sample(sample_size)
            
            game_data = []
            for i, (_, game) in enumerate(game_sample.iterrows()):
                game_data.append([i+1, game['id'], game['app_name']])
            
            print(f"Showing a random sample of {sample_size} games:")
            print(tabulate(game_data, headers=["#", "Game ID", "Game Name"], tablefmt="pretty"))
            
            return game_sample
    except Exception as e:
        print(f"Error listing games: {e}")
        return pd.DataFrame()
    
def evaluate_all_models():
    """Run evaluation on all recommendation models"""
    print("\n=== Evaluating All Recommendation Models ===")
    
    # Define k values being evaluated
    k_values = [5, 10, 20]
    results = {}
    
    # Evaluate ALS
    print("\n--- Evaluating ALS ---")
    try:
        als_results = evaluate_als()
        results['ALS'] = als_results
    except Exception as e:
        print(f"Error evaluating ALS: {e}")
    
    # Evaluate item-based CF
    print("\n--- Evaluating Item-based CF ---")
    try:
        item_cf_results = evaluate_item_based_cf()
        results['Item-based CF'] = item_cf_results
    except Exception as e:
        print(f"Error evaluating Item-based CF: {e}")
    
    # Evaluate user-based CF
    print("\n--- Evaluating User-based CF ---")
    try:
        user_cf_results = evaluate_user_based_cf()
        results['User-based CF'] = user_cf_results
    except Exception as e:
        print(f"Error evaluating User-based CF: {e}")
    
    # Print summary
    if results:
        print("\n=== Evaluation Summary ===")
        
        # Print header
        header = "Model"
        for k in k_values:
            header += f" | Precision@{k} | Recall@{k} | F1@{k}"
        print(header)
        print("-" * len(header))
        
        # Print results
        for model_name, model_results in results.items():
            row = model_name
            for k in k_values:
                if k in model_results:
                    precision, recall, f1 = model_results[k]
                    row += f" | {precision:.4f} | {recall:.4f} | {f1:.4f}"
                else:
                    row += " | N/A | N/A | N/A"
            print(row)
    
def test_recommendations(user_id=None, game_id=None):
    """Test all recommendation methods with a sample user and game"""
    print("\n=== Testing Recommendations ===")
    
    # If user_id is not provided, get one
    if user_id is None:
        sample_users = list_available_users()
        if sample_users:
            user_id = sample_users[0]
        else:
            print("Could not find a suitable user for testing")
            return
    
    # If game_id is not provided, get one
    if game_id is None:
        game_sample = list_available_games()
        if not game_sample.empty:
            game_id = game_sample.iloc[0]['id']
        else:
            print("Could not find a suitable game for testing")
            return
    
    print(f"Testing recommendations for User ID: {user_id} and Game ID: {game_id}")
    
    # Simple helper to catch errors
    def safe_recommend(name, func):
        print(f"\n--- {name} ---")
        try:
            start_time = time.time()
            results = func()
            elapsed = time.time() - start_time
            print(f"Execution time: {elapsed:.2f} seconds")
            print(results.head(5))
        except Exception as e:
            print(f"Error: {e}")
    
    # Test user-based recommendations
    safe_recommend("ALS Recommendations", lambda: get_als_recommendations(user_id))
    safe_recommend("Item-based CF Recommendations", lambda: get_item_based_cf_recommendations(user_id))
    safe_recommend("User-based CF Recommendations", lambda: get_user_based_cf_recommendations(user_id))
    
    # Test item-based recommendations
    safe_recommend("Content-based Recommendations", lambda: get_content_based_recommendations(game_id))
    safe_recommend("Text-based Recommendations", lambda: get_text_based_recommendations(game_id))
    safe_recommend("Association Rule Recommendations", lambda: get_association_recommendations(game_id))
    
    # Test non-personalized recommendations
    safe_recommend("Trending Recommendations", lambda: get_trending_recommendations())

def main():
    """Main function to run the recommendation system"""
    parser = argparse.ArgumentParser(description="Game Recommendation System")
    parser.add_argument("--check", action="store_true", help="Check and prepare prerequisites")
    parser.add_argument("--train-als", action="store_true", help="Train ALS model")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing data")
    parser.add_argument("--list-users", action="store_true", help="List sample users for testing")
    parser.add_argument("--list-games", action="store_true", help="List sample games for testing")
    parser.add_argument("--test", action="store_true", help="Test recommendation methods")
    parser.add_argument("--user", type=str, help="User ID for testing recommendations")
    parser.add_argument("--game", type=float, help="Game ID for testing recommendations")
    parser.add_argument("--clear-cache", action="store_true", help="Clear data cache before running")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all recommendation models")
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        config.clear_cache()
        print("Data cache cleared")
    
    # Ensure data directories exist
    ensure_data_directories()
    
    # Process commands
    if args.list_users:
        list_available_users()
    
    if args.list_games:
        list_available_games()
    
    if args.train_als:
        train_als_model(force=args.force)
    
    if args.test or args.user or args.game:
        test_recommendations(args.user, args.game)
    
    if args.evaluate:
        evaluate_all_models()
    
    # If no arguments provided, show usage
    if not any([args.check, args.train_als, args.list_users, args.list_games, 
                args.test, args.user, args.game, args.clear_cache, args.force,
                args.evaluate]):
        parser.print_help()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")