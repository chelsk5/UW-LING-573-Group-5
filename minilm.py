# Author: Nina Koh
# MiniLM-based ante-hoc selectors using sentence-transformers/all-MiniLM-L6-v2

import os
import json
import glob
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {}  # maps topics to their corresponding list of reviews
    search_path = os.path.join(data_dir, "*.txt.data") # data_dir is "topics/"
    files = glob.glob(search_path)
    
    for file in files:
        filename = os.path.basename(file)
        base = os.path.splitext(filename)[0]
        topic = os.path.splitext(base)[0]
        
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for line in f:
                if line.strip():
                    cleaned_line = clean_line(line) # clean line
                    lines.append(cleaned_line)
            data[topic] = lines
    
    return data

def clean_line(line):
    """Clean the given line by removing line numbers and extra whitespace"""
    # Remove line numbers (e.g., "1|" at the beginning)
    line = re.sub(r'^\d+\|', '', line)
    line = line.strip()
    return line

def encode_reviews(model, reviews):
    """Encode a list of reviews into embeddings using the MiniLM model"""
    return model.encode(reviews)

def compute_centroid(embeddings):
    """Compute the mean (centroid) of review embeddings"""
    return np.mean(embeddings, axis=0)

def simple_semantic_ranking(embeddings, centroid, k=3):
    """
    Compute cosine similarity between each review embedding and the centroid,
    and pick the top-k reviews.
    
    Args:
        embeddings: numpy array of review embeddings
        centroid: numpy array representing the centroid
        k: number of top reviews to select (default: 3)
    
    Returns:
        List of indices of selected reviews
    """
    # Reshape centroid for cosine similarity computation
    centroid = centroid.reshape(1, -1)
    
    # Compute cosine similarities
    similarities = cosine_similarity(embeddings, centroid).flatten()
    
    # Get indices of top-k most similar reviews
    top_k_indices = np.argsort(similarities)[-k:][::-1]  # Sort in descending order
    
    return top_k_indices.tolist()

def mmr_semantic_diversity(embeddings, centroid, k=3, n_candidates=10, lambda_param=0.6):
    """
    Run MMR on the top-N candidates to pick the final top-k reviews.
    Combines semantic relevance with diversity.
    
    Args:
        embeddings: numpy array of review embeddings
        centroid: numpy array representing the centroid
        k: number of final reviews to select (default: 3)
        n_candidates: number of top candidates to consider (default: 10)
        lambda_param: balance between relevance and diversity (default: 0.6)
    
    Returns:
        List of indices of selected reviews
    """
    # First, get top-N candidates using simple semantic ranking
    top_n_indices = simple_semantic_ranking(embeddings, centroid, n_candidates)
    
    if len(top_n_indices) <= k:
        return top_n_indices
    
    # Get embeddings for top-N candidates
    candidate_embeddings = embeddings[top_n_indices]
    
    # Reshape centroid for cosine similarity computation
    centroid = centroid.reshape(1, -1)
    
    # Compute relevance scores (cosine similarity to centroid)
    relevance_scores = cosine_similarity(candidate_embeddings, centroid).flatten()
    
    # Compute pairwise similarities between candidates
    sim_matrix = cosine_similarity(candidate_embeddings)
    
    # Initialize MMR selection
    selected = []
    remaining = list(range(len(top_n_indices)))
    
    # Iteratively select reviews based on MMR
    while len(selected) < k and remaining:
        mmr_scores = []
        
        for idx in remaining:
            # Compute redundancy penalty (max similarity to already selected)
            redundancy = np.max(sim_matrix[idx, selected]) if selected else 0
            
            # MMR = λ * relevance - (1 - λ) * redundancy
            mmr_score = lambda_param * relevance_scores[idx] - (1 - lambda_param) * redundancy
            mmr_scores.append(mmr_score)
        
        # Select the review with the highest MMR score
        max_mmr_index = np.argmax(mmr_scores)
        selected.append(remaining[max_mmr_index])
        remaining.remove(remaining[max_mmr_index])
    
    # Map back to original indices
    return [top_n_indices[i] for i in selected]

def run_minilm_selectors(data_dir="topics/", output_dir="minilm_results/"):
    """
    Run both MiniLM ante-hoc selectors on all topics and save results.
    
    Args:
        data_dir: directory containing .txt.data files
        output_dir: directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the MiniLM model
    print("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load data
    print("Loading review data...")
    data = load_data(data_dir)
    
    # Initialize results dictionaries
    simple_rankings = {}
    mmr_rankings = {}
    
    print(f"Processing {len(data)} topics...")
    
    for topic, reviews in data.items():
        print(f"Processing topic: {topic}")
        
        if len(reviews) < 3:
            print(f"  Warning: Topic {topic} has only {len(reviews)} reviews, skipping...")
            continue
        
        # Encode reviews
        embeddings = encode_reviews(model, reviews)
        
        # Compute centroid
        centroid = compute_centroid(embeddings)
        
        # Method 1: Simple semantic ranking
        simple_indices = simple_semantic_ranking(embeddings, centroid, k=3)
        simple_selected = [reviews[i] for i in simple_indices]
        simple_rankings[topic] = simple_selected
        
        # Method 2: MMR with semantic + diversity
        mmr_indices = mmr_semantic_diversity(embeddings, centroid, k=3, n_candidates=10, lambda_param=0.6)
        mmr_selected = [reviews[i] for i in mmr_indices]
        mmr_rankings[topic] = mmr_selected
    
    # Save results
    simple_output_path = os.path.join(output_dir, "minilm_simple_rankings.json")
    mmr_output_path = os.path.join(output_dir, "minilm_mmr_rankings.json")
    
    with open(simple_output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_rankings, f, indent=2, ensure_ascii=False)
    
    with open(mmr_output_path, 'w', encoding='utf-8') as f:
        json.dump(mmr_rankings, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}")
    print(f"Simple semantic rankings: {simple_output_path}")
    print(f"MMR semantic+diversity rankings: {mmr_output_path}")

if __name__ == "__main__":
    run_minilm_selectors()
