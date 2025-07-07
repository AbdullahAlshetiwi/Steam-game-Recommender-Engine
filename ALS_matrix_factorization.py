import numpy as np
import pandas as pd
import pickle
import os
import time

def load_data():
    """Load matrix and create dictionaries"""
    matrix_path = 'processing/processedData/user_item_matrix.pkl'
    print(f"Loading matrix from {matrix_path}")
    
    user_item_matrix = pd.read_pickle(matrix_path)
    
    # Create dictionaries
    user2game = {}
    game2user = {}
    user_game2rating = {}
    
    for user in user_item_matrix.index:
        owned_games = user_item_matrix.columns[user_item_matrix.loc[user] == True].tolist()
        user2game[user] = owned_games
        
        for game in owned_games:
            if game not in game2user:
                game2user[game] = []
            game2user[game].append(user)
            user_game2rating[(user, game)] = 1.0
    
    # Create indices
    users = list(user2game.keys())
    games = list(game2user.keys())
    user_idx = {user: i for i, user in enumerate(users)}
    game_idx = {game: i for i, game in enumerate(games)}
    
    # Global mean (ownership probability)
    mu = len(user_game2rating) / (len(users) * len(games))
    
    return user_item_matrix, user2game, game2user, user_game2rating, users, games, user_idx, game_idx, mu

def run_als(user2game, game2user, user_game2rating, users, games, user_idx, game_idx, mu, k=100, reg=0.1, iterations=25):
    """Run ALS algorithm"""
    # Initialize matrices
    n_users = len(users)
    n_games = len(games)
    
    print(f"Initializing matrices: Users={n_users}, Games={n_games}, Factors={k}")
    np.random.seed(42)
    U = np.random.normal(0, 0.1, (n_users, k))
    V = np.random.normal(0, 0.1, (n_games, k))
    b = np.zeros(n_users)
    c = np.zeros(n_games)
    
    # Main ALS iterations
    for it in range(iterations):
        print(f"Iteration {it + 1}/{iterations}")
        
        # Update user features
        for i, user in enumerate(users):
            u_idx = user_idx[user]
            sum1 = np.zeros((k, k))
            sum2 = np.zeros(k)
            
            for game in user2game[user]:
                m_idx = game_idx[game]
                r = user_game2rating[(user, game)]
                v_j = V[m_idx].reshape(-1, 1)
                sum1 += np.dot(v_j, v_j.T)
                sum2 += (r - b[u_idx] - c[m_idx] - mu) * V[m_idx]
            
            sum1 += reg * np.eye(k)
            U[u_idx] = np.linalg.solve(sum1, sum2)
        
        # Update game features
        for i, game in enumerate(games):
            m_idx = game_idx[game]
            sum1 = np.zeros((k, k))
            sum2 = np.zeros(k)
            
            for user in game2user[game]:
                u_idx = user_idx[user]
                r = user_game2rating[(user, game)]
                u_i = U[u_idx].reshape(-1, 1)
                sum1 += np.dot(u_i, u_i.T)
                sum2 += (r - b[u_idx] - c[m_idx] - mu) * U[u_idx]
            
            sum1 += reg * np.eye(k)
            V[m_idx] = np.linalg.solve(sum1, sum2)
        
        # Update user biases
        for i in range(len(b)):
            user = users[i]
            games_rated = user2game[user]
            x = 1/(len(games_rated) + reg)
            sum_val = 0
            
            for game in games_rated:
                j = game_idx[game]
                r = user_game2rating[(user, game)]
                sum_val += r - np.dot(U[i], V[j]) - c[j] - mu
            
            b[i] = x*sum_val
        
        # Update game biases
        for j in range(len(c)):
            game = games[j]
            users_rated = game2user[game]
            x = 1/(len(users_rated) + reg)
            sum_val = 0
            
            for user in users_rated:
                i = user_idx[user]
                r = user_game2rating[(user, game)]
                sum_val += r - np.dot(U[i], V[j]) - b[i] - mu
            
            c[j] = x*sum_val
    
    return U, V, b, c

def save_model(U, V, b, c, user_idx, game_idx):
    """Save model to disk"""
    os.makedirs('processing/processedData', exist_ok=True)
    
    with open('processing/processedData/U_matrix.pkl', 'wb') as f:
        pickle.dump(U, f)
    with open('processing/processedData/V_matrix.pkl', 'wb') as f:
        pickle.dump(V, f)
    with open('processing/processedData/user_biases.pkl', 'wb') as f:
        pickle.dump(b, f)
    with open('processing/processedData/game_biases.pkl', 'wb') as f:
        pickle.dump(c, f)
    with open('processing/processedData/user_idx.pkl', 'wb') as f:
        pickle.dump(user_idx, f)
    with open('processing/processedData/game_idx.pkl', 'wb') as f:
        pickle.dump(game_idx, f)

def load_model():
    """Load model from disk"""
    with open('processing/processedData/U_matrix.pkl', 'rb') as f:
        U = pickle.load(f)
    with open('processing/processedData/V_matrix.pkl', 'rb') as f:
        V = pickle.load(f)
    with open('processing/processedData/user_biases.pkl', 'rb') as f:
        b = pickle.load(f)
    with open('processing/processedData/game_biases.pkl', 'rb') as f:
        c = pickle.load(f)
    with open('processing/processedData/user_idx.pkl', 'rb') as f:
        user_idx = pickle.load(f)
    with open('processing/processedData/game_idx.pkl', 'rb') as f:
        game_idx = pickle.load(f)
    
    return U, V, b, c, user_idx, game_idx

def predict(user, games, U, V, b, c, mu, user_idx, game_idx, user2game, game2user):
    """Predict ownership probability for a user across games"""
    predictions = []
    u_idx = user_idx[user]
    
    # Get pure ALS predictions
    for game in games:
        m_idx = game_idx[game]
        
        # Base ALS prediction
        als_pred = np.dot(U[u_idx], V[m_idx]) + b[u_idx] + c[m_idx] + mu
        
        # Convert to probability with sigmoid
        prob = 1 / (1 + np.exp(-als_pred))
        predictions.append((game, prob))
    
    # Sort predictions
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    return sorted_predictions

def recommend_games(user, user2game, game2user, games, U, V, b, c, mu, user_idx, game_idx, top_n=10):
    """Generate game recommendations for a user"""
    # Get games the user doesn't own
    owned_games = set(user2game[user])
    candidate_games = list(set(games) - owned_games)
    
    # Predict scores for candidate games
    predictions = predict(user, candidate_games, U, V, b, c, mu, user_idx, game_idx, user2game, game2user)
    
    return predictions[:top_n]