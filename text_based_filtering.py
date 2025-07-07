import pandas as pd
import numpy as np
from config import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_game_reviews():

    game_reviews_df = pd.DataFrame(config.game_names_id)
    game_reviews_df['reviews'] = ' '
    for i in range(len(config.reviews)):
        # Print progress every 500 reviews
        if i % 500 == 0:
            print(f"Processing review {i}/{len(config.reviews)} ({(i/len(config.reviews)*100):.1f}%)")
        # Get the reviews string for the current game
        # Skip if reviews are null/missing
        if config.reviews.isnull().loc[i]['reviews']:
            continue

        user_reviews = config.reviews.loc[i]['reviews']
        for j in range (len(user_reviews)):
            review = user_reviews[j].get('review')
            id = float(user_reviews[j].get('item_id'))
            if game_reviews_df.loc[game_reviews_df['id'] == id]['reviews'].values.size == 0:
                continue
            game_reviews_df.loc[game_reviews_df['id'] == id, 'reviews'] = game_reviews_df.loc[game_reviews_df['id'] == id, 'reviews'].values[0] + review + ' '

    game_reviews_df = game_reviews_df[game_reviews_df['reviews'] != ' ']

    pd.to_pickle(game_reviews_df, 'processing/processedData/game_reviews.pkl')


def process_tfidf():
    """
    Process the game reviews using TF-IDF and calculate cosine similarity.
    """
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.7, stop_words='english')
        
    vectorized_data = vectorizer.fit_transform(config.game_reviews['reviews'])

    tfidf_df = pd.DataFrame(
        vectorized_data.toarray(), 
        columns=vectorizer.get_feature_names_out()
    )

    tfidf_df.index = config.game_reviews['id']
    tfidf_df.head() 

    cosine_similarity_array = cosine_similarity(tfidf_df)

    cosine_similarity_df = pd.DataFrame(
        cosine_similarity_array, 
        index=tfidf_df.index, 
        columns=tfidf_df.index
    )

    pd.to_pickle(cosine_similarity_df, 'processing/processedData/tfidf_similarity_matrix.pkl')
