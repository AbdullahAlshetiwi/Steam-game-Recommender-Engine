import numpy as np
import pandas as pd
import os
import cupy as cp
import pickle
import time
import ast
from config import config
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('.')  # Add current directory to path
from content_based_filtering import process_game_tags, compute_similarity_matrix_gpu
from text_based_filtering import process_game_reviews, process_tfidf
from non_personalized_trending import process_game_ratings_popularity, generate_user_item_matrix, game_association_rule_mining

def get_top_k_similar_games_from_matrix(similarity_matrix, game_id, k):
    """
    Returns the top k most similar games to the specified game.
    """
    if game_id not in similarity_matrix.index:
        raise ValueError(f"Game with ID {game_id} not found in the similarity matrix")
    
    # Get similarities for the specified game
    similarities = similarity_matrix.loc[game_id].reset_index()
    similarities.columns = ['game_id', 'similarity']
    # Remove self-similarity
    similarities = similarities[similarities['game_id'] != game_id]

    # Sort by similarity and return top k
    similarities = similarities.sort_values('similarity', ascending=False).head(k)
    
    # Create a dictionary to map IDs to names - but only for the top k games we're actually showing
    id_list = similarities['game_id'].tolist()
    names_dict = {}
    
    for game_id in id_list:
        match = config.game_names_id[config.game_names_id['id'].astype(float) == float(game_id)]
                
        # If we found a match, use it
        if not match.empty:
            names_dict[game_id] = match.iloc[0, 0]  # Get the name from column 0
        else:
            names_dict[game_id] = f"Unknown Game (ID: {game_id})"
            print(f"Debug: Could not find game with ID {game_id} (type: {type(game_id)})")
            # Print first few entries of game_names_id for debugging
            print(f"Debug: First few entries of game_names_id: {config.game_names_id.head()}")
            
    
    # Create result DataFrame
    result = pd.DataFrame()
    result['game_name'] = similarities['game_id'].map(names_dict)
    result['similarity'] = similarities['similarity']
    
    return result

def get_content_based_recommendations(game_id, k=10):
    """
    Get content-based recommendations using game tags similarity.
    
    Args:
        game_id (int): ID of the target game
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games and similarity scores
    """
    # Check if similarity matrix exists
    if not os.path.exists('processing/processedData/similarity_matrix.pkl'):
        print("Content-based similarity matrix not found. Generating now...")
        game_tags = process_game_tags()
        compute_similarity_matrix_gpu(game_tags)
    
    # Load similarity matrix
    sim_matrix = pd.read_pickle('processing/processedData/similarity_matrix.pkl')
    
    return get_top_k_similar_games_from_matrix(sim_matrix, game_id, k)

def get_text_based_recommendations(game_id, k=10):
    """
    Get text-based recommendations using review TF-IDF similarity.
    
    Args:
        game_id (int): ID of the target game
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games and similarity scores
    """
    # Check if TF-IDF similarity matrix exists
    if not os.path.exists('processing/processedData/tfidf_similarity_matrix.pkl'):
        print("TF-IDF similarity matrix not found. Generating now...")
        if not os.path.exists('processing/processedData/game_reviews.pkl'):
            process_game_reviews()
        process_tfidf()
    
    # Load TF-IDF similarity matrix
    tfidf_sim_matrix = pd.read_pickle('processing/processedData/tfidf_similarity_matrix.pkl')
    
    return get_top_k_similar_games_from_matrix(tfidf_sim_matrix, game_id, k)

def get_trending_recommendations(k=10):
    """
    Get non-personalized trending game recommendations based on Bayesian ratings.
    
    Args:
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with trending games and their Bayesian ratings
    """
    # Check if ratings data exists
    if not os.path.exists('processing/processedData/game_ratings_popularity.pkl'):
        print("Game ratings data not found. Generating now...")
        process_game_ratings_popularity()
    
    # Load the game ratings and popularity data
    game_ratings = pd.read_pickle('processing/processedData/game_ratings_popularity.pkl')
    
    # Return the top k games by Bayesian rating
    trending_games = game_ratings[['id', 'app_name', 'bayesian_rating', 'reviews_count']].head(k)
    
    # Create result DataFrame in the expected format
    result = pd.DataFrame()
    result['game_name'] = trending_games['app_name']
    result['bayesian_rating'] = trending_games['bayesian_rating']
    result['reviews_count'] = trending_games['reviews_count']
    
    return result

def get_association_recommendations(game_id, k=10):
    """
    Get recommendations using association rules (collaborative filtering approach).
    
    Args:
        game_id (int): ID of the target game
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games based on association rules
    """
    # Check if association rules exist
    if not os.path.exists('processing/processedData/association_rules.pkl'):
        print("Association rules not found. Generating now...")
        if not os.path.exists('processing/processedData/user_item_matrix.pkl'):
            generate_user_item_matrix()
        game_association_rule_mining()
    
    # Get game name from game_id using the same robust lookup method
    found_name = None
    
        
    match = config.game_names_id[config.game_names_id['id'].astype(float) == float(game_id)]
    
    if not match.empty:
        found_name = match.iloc[0, 0]
    else:
        raise ValueError(f"Game with ID {game_id} not found")
    
    game_name = found_name
    print(f"Looking for association rules for game: {game_name}")
    
    # Load association rules
    rules = pd.read_pickle('processing/processedData/association_rules.pkl')
    
    # Find rules where the antecedent contains our target game
    game_rules = []
    for i, row in rules.iterrows():
        antecedents = row['antecedents']
        # Check if game_name is in the frozen set of antecedents
        if game_name in antecedents:
            for consequent in row['consequents']:
                # Only add if not the same as our target game
                if consequent != game_name:
                    game_rules.append({
                        'game_name': consequent,
                        'confidence': row['confidence'],
                        'lift': row['lift']
                    })
    
    # Convert to DataFrame, sort by confidence, and return top k
    if not game_rules:
        print(f"No association rules found for game {game_name}")
        return pd.DataFrame(columns=['game_name', 'confidence', 'lift'])
    
    result = pd.DataFrame(game_rules)
    result = result.sort_values('confidence', ascending=False).head(k)
    
    return result[['game_name', 'confidence', 'lift']]
# Add these functions to utils.py

def get_als_recommendations(user_id, k=10):
    """
    Get ALS-based recommendations for a user.
    
    Args:
        user_id (str): ID of the target user
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games and their scores
    """
    import pickle
    import numpy as np
    import pandas as pd
    import os
    from ALS_matrix_factorization import load_model, predict, load_data
    
    # Check if ALS model files exist
    model_files = [
        'processing/processedData/U_matrix.pkl',
        'processing/processedData/V_matrix.pkl',
        'processing/processedData/user_biases.pkl',
        'processing/processedData/game_biases.pkl',
        'processing/processedData/user_idx.pkl',
        'processing/processedData/game_idx.pkl'
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        print("ALS model files not found.")
        raise FileNotFoundError("ALS model files not found. Please run ALS_matrix_factorization.py first.")
    
    # Load model and data
    U, V, b, c, user_idx, game_idx = load_model()
    _, user2game, game2user, user_game2rating, users, games, _, _, mu = load_data()
    
    # Check if user exists
    if user_id not in user_idx:
        raise ValueError(f"User with ID {user_id} not found in the model")
    
    # Get recommendations
    recommendations = predict(user_id, games, U, V, b, c, mu, user_idx, game_idx, user2game, game2user)
    recommendations = recommendations[:k]
    
    # Create result DataFrame
    result = pd.DataFrame()
    result['game_name'] = [game for game, _ in recommendations]
    result['score'] = [score for _, score in recommendations]
    
    return result

def get_item_based_cf_recommendations(user_id, k=10):
    """
    Get item-based collaborative filtering recommendations for a user.
    
    Args:
        user_id (str): ID of the target user
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games and their scores
    """
    import pickle
    import pandas as pd
    import os
    from user_item_based_cf import item_based_cf
    
    # Check if required files exist
    required_files = [
        'processing/processedData/user2game_dict.pkl',
        'processing/processedData/game2user_dict.pkl',
        'processing/processedData/game_similarity_matrix.pkl'
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Required files for item-based CF not found.")
        raise FileNotFoundError("Required files for item-based CF not found. Please run user_item_based_cf.py first.")
    
    # Load data
    with open('processing/processedData/user2game_dict.pkl', 'rb') as f:
        user2game = pickle.load(f)
    with open('processing/processedData/game2user_dict.pkl', 'rb') as f:
        game2user = pickle.load(f)
    with open('processing/processedData/game_similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    
    # Check if user exists
    if user_id not in user2game:
        raise ValueError(f"User with ID {user_id} not found in the data")
    
    # Get recommendations
    recommendations = item_based_cf(user_id, user2game, game2user, similarity_matrix, top_n=k)
    
    # Create result DataFrame
    result = pd.DataFrame()
    result['game_name'] = [game for game, _ in recommendations]
    result['score'] = [score for _, score in recommendations]
    
    return result

def get_user_based_cf_recommendations(user_id, k=10):
    """
    Get user-based collaborative filtering recommendations for a user.
    
    Args:
        user_id (str): ID of the target user
        k (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended games and their scores
    """
    import pickle
    import pandas as pd
    import os
    from user_item_based_cf import user_based_cf
    
    # Check if required files exist
    required_files = [
        'processing/processedData/user2game_dict.pkl',
        'processing/processedData/game2user_dict.pkl'
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Required files for user-based CF not found.")
        raise FileNotFoundError("Required files for user-based CF not found. Please run user_item_based_cf.py first.")
    
    # Load data
    with open('processing/processedData/user2game_dict.pkl', 'rb') as f:
        user2game = pickle.load(f)
    with open('processing/processedData/game2user_dict.pkl', 'rb') as f:
        game2user = pickle.load(f)
    
    # Check if user exists
    if user_id not in user2game:
        raise ValueError(f"User with ID {user_id} not found in the data")
    
    # Get recommendations
    recommendations = user_based_cf(user_id, user2game, game2user, top_n=k)
    
    # Create result DataFrame
    result = pd.DataFrame()
    result['game_name'] = [game for game, _ in recommendations]
    result['score'] = [score for _, score in recommendations]
    
    return result
