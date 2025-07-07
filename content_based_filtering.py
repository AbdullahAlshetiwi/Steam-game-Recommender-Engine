from config import config
import pandas as pd
import numpy as np
import ast
import time
import cupy as cp

def process_game_tags():
    """
    Process game tags from the games dataframe and create a one-hot encoded matrix.
    This function extracts all unique tags from the games dataset, then creates a binary
    matrix where each row represents a game and each column represents whether the game
    has a specific tag (1) or not (0).
    
    Returns:
        pd.DataFrame: A DataFrame with game IDs as indices and tags as columns, with binary values.
    """
    start_time = time.time()
    
    # Step 1: Extract all unique tags from the games dataset
    print("Starting tag extraction process...")
    tags = []
    total_games = len(config.games) - 1
    
    for i in range(1, len(config.games)):
        # Print progress every 500 games
        if i % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Processing game {i}/{total_games} ({(i/total_games*100):.1f}%). Elapsed time: {elapsed:.2f}s")
            
        # Get the tags string for the current game
        tags_string = config.games.loc[i]['tags']
        
        # Skip if tags are null/missing
        if config.games.isnull().loc[i]['tags']:
            continue
            
        # Convert string representation of list to actual list
        tags_list = tags_string
        
        # Add any new tags to our master list
        for each in tags_list:
            if each not in tags:
                tags.append(each)
    
    # Report progress after tag extraction
    print(f"Extracted {len(tags)} unique tags from {total_games} games.")
    
    # Step 2: Create empty DataFrame with games as rows and tags as columns
    game_tags = pd.DataFrame(index=config.games['id'], columns=tags)
    print("Created empty tag matrix. Beginning to populate with game data...")
    
    # Step 3: Fill the DataFrame with 1s where a game has a specific tag
    process_start = time.time()
    for i in range(1, len(config.games)):
        # Print progress every 500 games
        if i % 500 == 0:
            elapsed = time.time() - process_start
            percentage = (i/total_games*100)
            games_per_second = i / (elapsed + 0.001)  # Avoid division by zero
            remaining = (total_games - i) / (games_per_second + 0.001)
            print(f"Populating matrix: {i}/{total_games} games ({percentage:.1f}%). Rate: {games_per_second:.1f} games/s. Est. remaining: {remaining:.1f}s")
        
        id = game_tags.index[i]
        
        # Skip if game ID doesn't exist in the dataset
        if config.games.loc[config.games['id'] == id]['tags'].empty:
            continue
            
        # Skip if tags are null/missing for this game
        if pd.isnull(config.games.loc[config.games['id'] == id]['tags'].values[0]):
            continue
            
        # Get tags for this game and convert to list
        tags_string = config.games.loc[config.games['id'] == id]['tags'].values[0]
        tags_list = ast.literal_eval(tags_string)
        
        # Mark each tag this game has with a 1
        for each in tags_list:
            game_tags.loc[id][each] = 1
    
    # Step 4: Fill all remaining NaN values with 0 (indicating absence of tag)
    game_tags.fillna(0, inplace=True)
    
    # Step 5: Save processed data (using CSV for this smaller file is still reasonable)
    output_path = 'processing/processedData/game_tags.csv'
    game_tags.to_csv(output_path)
    
    # Also save a pickle version for potentially faster loading
    pickle_path = 'processing/processedData/game_tags.pkl'
    game_tags.to_pickle(pickle_path)
    
    # Calculate and display processing statistics
    total_time = time.time() - start_time
    print(f"\nProcessing complete in {total_time:.2f} seconds.")
    print(f"Created tag matrix with {game_tags.shape[0]} games and {game_tags.shape[1]} tags.")
    print(f"Data saved to {output_path} and {pickle_path}")
    
    # Display sample of processed data (first 5 games, first 10 tags)
    if not game_tags.empty:
        sample_cols = min(10, game_tags.shape[1])
        print(f"\nSample of processed data (first 5 games, first {sample_cols} tags):")
        print(game_tags.iloc[:5, :sample_cols])
    
    return game_tags


def compute_similarity_matrix_gpu(game_tags_data):
    """
    Computes the full Jaccard similarity matrix for all games with optimized GPU processing.
    
    Args:
        game_tags_data (pd.DataFrame): DataFrame with game IDs as index and tags as columns (binary values)
        
    Returns:
        pd.DataFrame: A similarity matrix where both rows and columns are game IDs
    """
    start_time = time.time()
    
    # Get all game IDs
    all_game_ids = game_tags_data.index.values
    num_games = len(all_game_ids)
    print(f"Starting optimized similarity computation for {num_games} games...")
    
    # Get all game tags as a feature matrix
    all_game_tags = game_tags_data.values
    
    # Transfer to GPU once
    all_game_tags_gpu = cp.asarray(all_game_tags)
    
    # Initialize matrix in smaller blocks to avoid memory issues
    block_size = 1000  # Process in 1000×1000 game blocks
    num_blocks = (num_games + block_size - 1) // block_size
    
    # Empty matrix to store results
    similarity_matrix = np.zeros((num_games, num_games))
    
    # Process blocks of the matrix
    for block_i in range(num_blocks):
        i_start = block_i * block_size
        i_end = min((block_i + 1) * block_size, num_games)
        i_size = i_end - i_start
        
        # Create GPU arrays for intersection and union calculations
        games_i = all_game_tags_gpu[i_start:i_end]
        
        print(f"Processing block {block_i+1}/{num_blocks} (games {i_start}-{i_end-1})...")
        block_start = time.time()
        
        for block_j in range(block_i, num_blocks):
            j_start = block_j * block_size
            j_end = min((block_j + 1) * block_size, num_games)
            j_size = j_end - j_start
            
            games_j = all_game_tags_gpu[j_start:j_end]
            
            # VECTORIZED COMPUTATION - this is the key optimization
            # Reshape for broadcasting: (i_size, 1, tags) and (1, j_size, tags)
            games_i_reshaped = games_i.reshape(i_size, 1, -1)
            games_j_reshaped = games_j.reshape(1, j_size, -1)
            
            # Calculate intersection and union for all pairs at once
            intersection = cp.sum(cp.logical_and(games_i_reshaped, games_j_reshaped), axis=2)
            union = cp.sum(cp.logical_or(games_i_reshaped, games_j_reshaped), axis=2)
            
            # Calculate Jaccard similarity (avoiding division by zero)
            jaccard = cp.zeros_like(intersection, dtype=cp.float32)
            valid_idx = union > 0
            jaccard[valid_idx] = intersection[valid_idx] / union[valid_idx]
            
            # Transfer results to CPU
            jaccard_np = cp.asnumpy(jaccard)
            
            # Fill the similarity matrix (both upper and lower triangles)
            similarity_matrix[i_start:i_end, j_start:j_end] = jaccard_np
            if i_start != j_start:  # Don't duplicate the diagonal blocks
                similarity_matrix[j_start:j_end, i_start:i_end] = jaccard_np.T
            
            # Free memory
            del jaccard
            cp.get_default_memory_pool().free_all_blocks()
            
        block_end = time.time()
        print(f"  Block completed in {block_end - block_start:.2f} seconds")
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=all_game_ids, columns=all_game_ids)
    
    # Save as pickle instead of CSV (much more efficient for large matrices)
    output_path = 'processing/processedData/similarity_matrix.pkl'
    print(f"Saving similarity matrix to {output_path}...")
    save_start = time.time()
    similarity_df.to_pickle(output_path)
    save_end = time.time()
    print(f"Matrix saved in {save_end - save_start:.2f} seconds.")
    
    end_time = time.time()
    print(f"Similarity matrix computation completed in {end_time - start_time:.2f} seconds.")
    print(f"Matrix shape: {similarity_df.shape}")
    print(f"Data saved to {output_path}")
    
    return similarity_df