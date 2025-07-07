import pandas as pd
import numpy as np
from config import config
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pickle

def process_game_ratings_popularity():
    """
    Process the game ratings and popularity data.
    """
    
    # Create a DataFrame to hold the game ratings and popularity
    game_ratings_popularity = pd.DataFrame({
        'id': config.games['id'],
        'app_name': config.games['app_name'],
        'rating': config.games['sentiment'],
        'reviews_count': 0
    })

    # Convert Sentiment to numeric
    for i in range(len(game_ratings_popularity)):
        # Print progress every 500 reviews
        if i % 500 == 0:
            print(f"Processing rating {i}/{len(game_ratings_popularity)} ({(i/len(game_ratings_popularity)*100):.1f}%)")
        # Convert the rating to a numeric value
        if game_ratings_popularity.loc[i, 'rating'] == 'Overwhelmingly Negative':
            game_ratings_popularity.loc[i, 'rating'] = 0
        elif game_ratings_popularity.loc[i, 'rating'] == 'Mostly Negative':
            game_ratings_popularity.loc[i, 'rating'] = 1
        elif game_ratings_popularity.loc[i, 'rating'] == 'Negative':
            game_ratings_popularity.loc[i, 'rating'] = 2
        elif game_ratings_popularity.loc[i, 'rating'] == 'Mixed':
            game_ratings_popularity.loc[i, 'rating'] = 3
        elif game_ratings_popularity.loc[i, 'rating'] == 'Positive':
            game_ratings_popularity.loc[i, 'rating'] = 4
        elif game_ratings_popularity.loc[i, 'rating'] == 'Mostly Positive':
            game_ratings_popularity.loc[i, 'rating'] = 5
        elif game_ratings_popularity.loc[i, 'rating'] == 'Overwhelmingly Positive':
            game_ratings_popularity.loc[i, 'rating'] = 6
        else:
            # Delete the row if the rating is not recognized
            game_ratings_popularity.drop(index=i, inplace=True)


    # Calculate the ratings count for each game
    for i in range(len(config.reviews)):
        # Print progress every 500 reviews
        if i % 500 == 0:
            print(f"Processing review {i}/{len(config.reviews)} ({(i/len(config.reviews)*100):.1f}%)")
        
        # Get the reviews string for the current game
        user_reviews = config.reviews.loc[i]['reviews']
        for j in range(len(user_reviews)):
            id = float(user_reviews[j].get('item_id'))
            game_ratings_popularity.loc[game_ratings_popularity['id'] == id, 'reviews_count'] += 1

    # Set Review Count to 1 for games with no reviews
    game_ratings_popularity.loc[game_ratings_popularity['reviews_count'] == 0, 'reviews_count'] = 1

    # Bayesian Scoring Calculation
    global_avg = game_ratings_popularity['rating'].mean()
    c = game_ratings_popularity['reviews_count'].mean()
    game_ratings_popularity['bayesian_rating'] = (
        (game_ratings_popularity['reviews_count'] * game_ratings_popularity['rating']) + (c * global_avg)
    ) / (game_ratings_popularity['reviews_count'] + c)

    # Sort the DataFrame by ratings count in descending order
    game_ratings_popularity.sort_values(by='bayesian_rating', ascending=False, inplace=True)
    # Reset the index
    game_ratings_popularity.reset_index(drop=True, inplace=True)

    # Save the processed DataFrame to a pickle file
    pd.to_pickle(game_ratings_popularity, 'processing/processedData/game_ratings_popularity.pkl')

    print(game_ratings_popularity.head(10)) 

def generate_user_item_matrix():
    """
    Perform association rule mining on the game ratings and popularity data.
    """
    
    # Creating User Item Matrix (optimized approach)
    print('Starting optimized matrix creation')

    # First, create a dictionary to quickly look up game names by id
    game_id_to_name = dict(zip(config.games['id'], config.games['app_name']))

    # Initialize an empty list to collect all user-game pairs
    user_game_pairs = []

    # Process each user's items more efficiently
    for i in range(len(config.items)):
        if i % 500 == 0:
            print(f"Processing item {i}/{len(config.items)} ({(i/len(config.items)*100):.1f}%)")
        
        user_id = config.items.iloc[i]['user_id']
        user_items = config.items.iloc[i]['items']
        
        # Collect all valid game ids for this user
        for j in range(len(user_items)):
            item_id = user_items[j].get('item_id')
            if item_id in game_id_to_name:
                game_name = game_id_to_name[item_id]
                user_game_pairs.append((user_id, game_name))

    # Create a DataFrame from the collected pairs
    pairs_df = pd.DataFrame(user_game_pairs, columns=['user_id', 'game_name'])

    # Create a pivot table from the pairs (this is much faster than filling cell by cell)
    user_item_matrix = pd.crosstab(
        index=pairs_df['user_id'],
        columns=pairs_df['game_name'],
        values=1,
        aggfunc='first'  # Just take the first occurrence
    ).fillna(0).astype(bool)

    # Remove columns with all zeros (though there shouldn't be any with this approach)
    user_item_matrix = user_item_matrix.loc[:, (user_item_matrix != 0).any(axis=0)]

    # Save the user-item matrix to a pickle file
    pd.to_pickle(user_item_matrix, 'processing/processedData/user_item_matrix.pkl')
    print('Optimized matrix creation completed')

def game_association_rule_mining():
    """
    Perform association rule mining on the game ratings and popularity data.
    """
    # Faster than standard pickle loading
    with open('processing/processedData/user_item_matrix.pkl', 'rb') as f:
        user_item_matrix_pickle = pickle.load(f)

    user_item_matrix = pd.DataFrame(user_item_matrix_pickle)


    # Perform association rule mining
    frequent_itemsets = fpgrowth(user_item_matrix, min_support=0.05, use_colnames=True, verbose=2)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    # Save the rules to a pickle file
    pd.to_pickle(rules, 'processing/processedData/association_rules.pkl')
    print('Association rule mining completed')



    
    