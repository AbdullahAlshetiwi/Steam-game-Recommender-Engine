import pandas as pd
import pickle
import os
from functools import lru_cache

# Global cache for loaded data
_data_cache = {}

# Configure base paths - adjust if needed
DATA_DIR = 'data'
PROCESSED_DIR = 'processing/processedData'

def get_data_path(filename, is_processed=False):
    """Get the correct path for a data file based on its location"""
    base_dir = PROCESSED_DIR if is_processed else DATA_DIR
    return os.path.join(base_dir, filename)

@lru_cache(maxsize=32)
def load_data(data_name):
    """Lazy-load data only when requested"""
    if data_name in _data_cache:
        return _data_cache[data_name]
    
    print(f"Loading {data_name}...")  # Debug log
    
    if data_name == 'reviews':
        path = get_data_path('reviews.pkl')
        data = pd.read_pickle(path)
    elif data_name == 'games':
        path = get_data_path('games.pkl')
        data = pd.read_pickle(path)
    elif data_name == 'items':
        path = get_data_path('items.pkl')
        with open(path, 'rb') as f:
            items_pickle = pickle.load(f)
        data = pd.DataFrame(items_pickle)
    elif data_name == 'game_tags_data':
        path = get_data_path('game_tags.csv', is_processed=True)
        if os.path.exists(path):
            data = pd.read_csv(path, index_col=0)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'similarity_matrix':
        path = get_data_path('similarity_matrix.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'game_reviews':
        path = get_data_path('game_reviews.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'tfidf_similarity_matrix':
        path = get_data_path('tfidf_similarity_matrix.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'user_item_matrix':
        path = get_data_path('user_item_matrix.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'user2game_dict':
        path = get_data_path('user2game_dict.pkl', is_processed=True)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Warning: {path} does not exist")
            data = {}
    elif data_name == 'game2user_dict':
        path = get_data_path('game2user_dict.pkl', is_processed=True)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Warning: {path} does not exist")
            data = {}
    elif data_name == 'game_similarity_matrix':
        path = get_data_path('game_similarity_matrix.pkl', is_processed=True)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Warning: {path} does not exist")
            data = {}
    elif data_name == 'association_rules':
        path = get_data_path('association_rules.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    elif data_name == 'game_ratings_popularity':
        path = get_data_path('game_ratings_popularity.pkl', is_processed=True)
        if os.path.exists(path):
            data = pd.read_pickle(path)
        else:
            print(f"Warning: {path} does not exist")
            data = pd.DataFrame()
    else:
        raise ValueError(f"Unknown data requested: {data_name}")
    
    # Cache the data
    _data_cache[data_name] = data
    print(f"Loaded {data_name}")  # Debug log
    return data

def clear_cache():
    """Clear all cached data"""
    global _data_cache
    _data_cache = {}
    # Also clear the lru_cache
    load_data.cache_clear()
    print("Data cache cleared")

# Create a config instance for modules to import from
class Config:
    # Define properties as instance methods
    @property
    def reviews(self):
        return load_data('reviews')
    
    @property
    def games(self):
        return load_data('games')
    
    @property
    def items(self):
        return load_data('items')
    
    @property
    def game_tags_data(self):
        return load_data('game_tags_data')
    
    @property
    def game_names_id(self):
        games_data = load_data('games')
        return games_data.loc[:, ['app_name', 'id']]
    
    @property
    def similarity_matrix(self):
        return load_data('similarity_matrix')
    
    @property
    def game_reviews(self):
        return load_data('game_reviews')
    
    @property
    def tfidf_similarity_matrix(self):
        return load_data('tfidf_similarity_matrix')
    
    @property
    def user_item_matrix(self):
        return load_data('user_item_matrix')
    
    # Add methods
    def load_data(self, data_name):
        return load_data(data_name)
    
    def clear_cache(self):
        clear_cache()

# Create an instance of the config to import
config = Config()