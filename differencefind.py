import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import difflib

# Function to load text files from a directory
def load_text_files(text_dir):
    text_files = glob.glob(os.path.join(text_dir, "*.txt"))
    texts = []
    filenames = []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
            filenames.append(os.path.basename(text_file))
    return texts, filenames

# Function to compute embeddings and perform DBSCAN clustering
def cluster_texts_with_dbscan(texts, eps=0.1, min_samples=2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    
    return labels, embeddings

# Function to compute similarity scores
def compute_similarity_scores(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# Function to print differences between two texts
def print_differences(text1, text2, file1, file2):
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    differences = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
    if differences:
        print(f"Differences between {file1} and {file2}:")
        print("\n".join(differences))
        print("\n ***NO DUPLICATES***")
    else:
        print("\n !!! DUPLICATES PRESENT !!!")

# Main function to load texts, compute embeddings, cluster texts, and display results
def main(text_dir, eps=0.3, min_samples=2):
    start_time = time.time()
    # Step 1: Load text files for clustering
    texts, filenames = load_text_files(text_dir)
    
    # Step 2: Compute embeddings and cluster texts using DBSCAN
    labels, embeddings = cluster_texts_with_dbscan(texts, eps, min_samples)
    
    # Step 3: Compute similarity scores
    similarity_matrix = compute_similarity_scores(embeddings)
    
    # Print differences between texts with similarity score above 0.99
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((filenames[i], texts[i], embeddings[i]))
    
    for label, files in clusters.items():
        if len(files) > 1:
            print(f"\nDifferences in Cluster {label}:")
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    file1, text1, embedding1 = files[i]
                    file2, text2, embedding2 = files[j]
                    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                    if similarity > 0.8:
                        print(f"\nSimilarity Score between {file1} and {file2}: {similarity}")
                        print_differences(text1, text2, file1, file2)
                    else:
                        print(f"\nSimilarity Score between {file1} and {file2}: {similarity}")
                        print("\n *** NO DUPLICATES ***")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
# Example usage:
if __name__ == "__main__":
    text_dir = '/Users/I748970/Downloads/TEXTFILES'  # Directory containing text files for clustering
    main(text_dir)
