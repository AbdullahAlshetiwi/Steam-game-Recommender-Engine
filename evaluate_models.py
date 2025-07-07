import numpy as np
import pandas as pd
import pickle
import os
import time
import math
import random
from config import config

def train_test_split_by_user(user2game, test_ratio=0.1, min_ratings=10, max_users=1000, random_seed=42):
    """
    Split the data into training and test sets on a per-user basis.
    Only users with at least min_ratings are kept.
    For each eligible user, test_ratio percent of games are held out for the test set.
    Limits to max_users for faster evaluation.
    
    Returns:
        train_user2game: Dictionary with training data
        test_items: Dictionary with test data
    """
    # Check if cached split exists
    cache_path = 'processing/processedData/train_test_split.pkl'
    if os.path.exists(cache_path):
        print("Loading cached train/test split...")
        with open(cache_path, 'rb') as f:
            train_test_data = pickle.load(f)
            return train_test_data['train_user2game'], train_test_data['test_items']
    
    print(f"Creating train/test split: {min_ratings} min ratings, {test_ratio*100}% test items, max {max_users} users")
    
    # Make a copy of the original user2game dictionary for training
    train_user2game = {user: list(games) for user, games in user2game.items()}
    
    # Dictionary to store test items (user: list of games)
    test_items = {}
    
    # Find eligible users first
    eligible_users = []
    for user, games in user2game.items():
        if len(games) >= min_ratings:
            eligible_users.append(user)
    
    # Select a random subset of eligible users
    np.random.seed(random_seed)
    if len(eligible_users) > max_users:
        selected_users = np.random.choice(eligible_users, max_users, replace=False)
    else:
        selected_users = eligible_users
    
    # Process selected users
    for user in selected_users:
        owned_games = user2game[user]
        
        # Randomly select test_count games to hold out
        test_count = max(1, int(len(owned_games) * test_ratio))
        np.random.seed(random_seed + hash(user) % 1000)  # User-specific seed
        test_games = np.random.choice(owned_games, min(test_count, len(owned_games)), replace=False)
        
        # Store test items
        test_items[user] = test_games.tolist()
        
        # Remove test games from training data
        train_user2game[user] = [game for game in train_user2game[user] if game not in test_games]
    
    print(f"Created test set with {len(test_items)} users out of {len(eligible_users)} eligible users")
    
    # Cache the split
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train_user2game': train_user2game,
            'test_items': test_items
        }, f)
    
    return train_user2game, test_items

def compute_ranking_metrics(recommendations, test_items, k=10):
    """
    Compute precision, recall and F1 score for recommendations
    
    Args:
        recommendations: Dict mapping users to list of (game, score) tuples
        test_items: Dict mapping users to list of held-out games
        k: Number of top items to consider
    """
    precision_list = []
    recall_list = []
    f1_list = []
    
    for user, test_games in test_items.items():
        if user not in recommendations:
            continue
            
        # Get top k recommended games
        top_k_recs = [game for game, _ in recommendations[user][:k]]
        
        # Calculate hit count (items in both test set and recommendations)
        hits = len(set(top_k_recs) & set(test_games))
        
        # Calculate metrics
        precision = hits / k if k > 0 else 0
        recall = hits / len(test_games) if test_games else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Calculate averages
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_f1 = np.mean(f1_list) if f1_list else 0
    
    return avg_precision, avg_recall, avg_f1

def evaluate_als():
    """Evaluate ALS recommendations at multiple K values"""
    print("===== Evaluating ALS recommendations =====")
    
    from processing.ALS_matrix_factorization import load_model, load_data
    from processing.utils import get_als_recommendations
    
    # Check if model files exist
    model_files = [
        'processing/processedData/U_matrix.pkl',
        'processing/processedData/V_matrix.pkl',
        'processing/processedData/user_biases.pkl',
        'processing/processedData/game_biases.pkl',
        'processing/processedData/user_idx.pkl',
        'processing/processedData/game_idx.pkl'
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        print("Error: ALS model files not found.")
        return
    
    # Load model and data using existing functions
    U, V, b, c, user_idx, game_idx = load_model()
    user_item_matrix, user2game, game2user, user_game2rating, users, games, _, _, mu = load_data()
    
    # Create train/test split
    train_user2game, test_items = train_test_split_by_user(user2game)
    
    # Define K values to evaluate
    k_values = [5, 10, 20]
    
    # Store results for each K
    results_by_k = {k: [] for k in k_values}
    
    # Generate recommendations for test users
    recommendations = {}
    test_users = list(test_items.keys())
    
    for i, test_user in enumerate(test_users):
        if test_user not in user_idx:
            continue
        
        # Get candidate games (all games excluding training games)
        candidate_games = [game for game in games if game not in train_user2game.get(test_user, [])]
        
        # Generate predictions
        predictions = []
        for game in candidate_games:
            if game in game_idx:
                m_idx = game_idx[game]
                u_idx = user_idx[test_user]
                
                # Raw prediction
                pred = np.dot(U[u_idx], V[m_idx]) + b[u_idx] + c[m_idx] + mu
                
                # Convert to probability
                prob = 1 / (1 + np.exp(-pred))
                predictions.append((game, prob))
        
        # Sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations[test_user] = predictions
        
        # Progress update
        if (i+1) % 10 == 0:
            print(f"ALS: Processed {i+1}/{len(test_users)} test users")
    
    # Evaluate at each K
    avg_results = {}
    for k in k_values:
        precision, recall, f1 = compute_ranking_metrics(recommendations, test_items, k)
        avg_results[k] = (precision, recall, f1)
        print(f"ALS - Average at K={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return avg_results

def evaluate_item_based_cf():
    """Evaluate item-based CF recommendations at multiple K values"""
    print("\n===== Evaluating item-based CF recommendations =====")
    
    from processing.user_item_based_cf import item_based_cf
    
    # Load data
    with open('processing/processedData/user2game_dict.pkl', 'rb') as f:
        user2game = pickle.load(f)
    with open('processing/processedData/game2user_dict.pkl', 'rb') as f:
        game2user = pickle.load(f)
    with open('processing/processedData/game_similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    
    # Create train/test split
    train_user2game, test_items = train_test_split_by_user(user2game)
    
    # Define K values to evaluate
    k_values = [5, 10, 20]
    
    # Generate recommendations for test users
    recommendations = {}
    test_users = list(test_items.keys())
    
    for i, test_user in enumerate(test_users):
        # Use the item_based_cf function from user_item_based_cf.py
        # We need to create a temporary version with just the training data
        temp_user2game = user2game.copy()
        temp_user2game[test_user] = train_user2game.get(test_user, [])
        
        # Get all predictions for this user
        preds = item_based_cf(test_user, temp_user2game, game2user, similarity_matrix, top_n=100)
        recommendations[test_user] = preds
        
        # Progress update
        if (i+1) % 10 == 0:
            print(f"Item-based CF: Processed {i+1}/{len(test_users)} test users")
    
    # Evaluate at each K
    avg_results = {}
    for k in k_values:
        precision, recall, f1 = compute_ranking_metrics(recommendations, test_items, k)
        avg_results[k] = (precision, recall, f1)
        print(f"Item-based CF - Average at K={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return avg_results

def evaluate_user_based_cf():
    """Evaluate user-based CF recommendations at multiple K values"""
    print("\n===== Evaluating user-based CF recommendations =====")
    
    from processing.user_item_based_cf import user_based_cf
    
    # Load data
    with open('processing/processedData/user2game_dict.pkl', 'rb') as f:
        user2game = pickle.load(f)
    with open('processing/processedData/game2user_dict.pkl', 'rb') as f:
        game2user = pickle.load(f)
    
    # Create train/test split
    train_user2game, test_items = train_test_split_by_user(user2game)
    
    # Define K values to evaluate
    k_values = [5, 10, 20]
    
    # Generate recommendations for test users
    recommendations = {}
    test_users = list(test_items.keys())
    
    for i, test_user in enumerate(test_users):
        # Use the user_based_cf function from user_item_based_cf.py
        # We need to create a temporary version with just the training data
        temp_user2game = user2game.copy()
        temp_user2game[test_user] = train_user2game.get(test_user, [])
        
        # Get all predictions for this user
        preds = user_based_cf(test_user, temp_user2game, game2user, top_n=100)
        recommendations[test_user] = preds
        
        # Progress update
        if (i+1) % 10 == 0:
            print(f"User-based CF: Processed {i+1}/{len(test_users)} test users")
    
    # Evaluate at each K
    avg_results = {}
    for k in k_values:
        precision, recall, f1 = compute_ranking_metrics(recommendations, test_items, k)
        avg_results[k] = (precision, recall, f1)
        print(f"User-based CF - Average at K={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return avg_results
