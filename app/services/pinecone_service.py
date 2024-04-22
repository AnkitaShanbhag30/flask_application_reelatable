import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np
from . import index

# Load environment variables
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face token is not set in environment variables")

# Initialize Sentence Transformer model
embeddings_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', hf_token)

# Constants
INDEX_DIMENSION = 1024  # Example dimension, set it to your index dimension
CHAR_LIMIT = 5000
keys_to_process = ['beliefs', 'desires', 'personality_traits', 'flaws']

def get_sparse_embeddings(traits_list, n_components=10):
    """Generates sparse embeddings for a list of trait and attribute strings."""
    embeddings = embeddings_model.encode(traits_list)

    if embeddings.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings

    pinecone_embeddings = []
    for embedding in reduced_embeddings:
        indices = list(range(len(embedding)))
        values = embedding.tolist()
        pinecone_embeddings.append({"indices": indices, "values": values})

    return pinecone_embeddings

def get_dense_embeddings(texts):
    pinecone_embeddings = embeddings_model.encode([t.replace("\n", " ") for t in texts], show_progress_bar=True)
    return pinecone_embeddings


def prepare_metadata(row):
    """Prepare metadata with detailed traits and truncated evidence."""
    def extract_traits(traits_dict, trait_name):
        """Extract and format traits from dictionaries."""
        traits_info = {}
        for idx, (key, value) in enumerate(traits_dict.items(), 1):
            trait_key = f"{trait_name}_{idx}_trait"
            evidence_key = f"{trait_name}_{idx}_evidence"
            traits_info[trait_key] = value['trait']
            evidence = value['evidence']
            if len(evidence) > CHAR_LIMIT:
                evidence = evidence[:CHAR_LIMIT]
            traits_info[evidence_key] = evidence
        return traits_info

    overview = row['overview']
    if len(overview) > CHAR_LIMIT:
        overview = overview[:CHAR_LIMIT]

    metadata = {
        'title': row['title'],
        'tags': row['tags'],
        'overview': overview,
        'genres': row['genres'].split(','),
        'protagonist': row['protagonist'],
        'year': int(row['year']),
        'votes': int(row['votes']),
        'rating': float(row['rating']),
        'popularity': float(row['popularity']),
        'poster_url': row['poster_url'],
    }

    for key in keys_to_process:
        metadata.update(extract_traits(row[key], key))

    return metadata

def hybrid_scale(dense, sparse, alpha):
    """
    Scale the sparse and dense vectors for hybrid search according to the alpha parameter.
    :param dense: Dense vector
    :param sparse: Sparse vector
    :param alpha: Weighting factor between 0 and 1 for dense vector contribution
    :return: Scaled dense and sparse vectors
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    return scaled_dense, scaled_sparse

def hybrid_query(vector, sparse_vector, top_k, include_metadata=True):
    """
    Query Pinecone with both dense and sparse vector embeddings.
    :param vector: Dense vector for the query
    :param sparse_vector: Sparse vector for the query
    :param top_k: Number of top results to return
    :param include_metadata: Whether to include metadata in the query results
    :return: Search results from Pinecone
    """
    query = {
        "vector": vector,
        "sparse_vector": sparse_vector,
        "top_k": top_k,
        "include_metadata": include_metadata
    }
    return index.query(**query)

def search_movies(traits, top_k=10, alpha=0.5):
    """
    Search movies by given traits using hybrid search.
    :param traits: List of traits to search for
    :param top_k: Number of results to return
    :param alpha: Weighting factor for dense vector contribution in hybrid search
    :return: List of movie recommendations
    """
    # Here you would implement logic similar to Pinecone's hybrid search example:
    # 1. Generate sparse embeddings from the traits
    # 2. Generate dense embeddings from the traits or related text
    # 3. Scale both embeddings using the alpha parameter
    # 4. Perform the hybrid query
    # Example placeholders for dense and sparse embeddings generation
    dense_embeddings = get_dense_embeddings(traits)
    sparse_embeddings = get_sparse_embeddings(traits)
    
    # Scale embeddings
    dense_vector, sparse_vector = hybrid_scale(dense_embeddings, sparse_embeddings, alpha)
    
    # Perform hybrid search query
    results = hybrid_query(dense_vector, sparse_vector, top_k)
    
    # Return results
    return [match['metadata'] for match in results['matches']]

import numpy as np
from . import index

def get_metadata_by_movie(movie_name, seed_vector=None):
    """
    Retrieves movie metadata for an exact title match from the Pinecone index.
    Optionally uses a seed vector for the query.
    
    :param movie_name: The exact title of the movie to search for.
    :param seed_vector: Optional numpy array representing a seed vector. If None, a random vector is generated.
    :return: Movie metadata if found, None otherwise.
    """
    if seed_vector is None:
        # Generate a random vector of the preset dimension (INDEX_DIMENSION)
        seed_vector = np.random.rand(INDEX_DIMENSION).tolist()
    
    # Define the query to find the movie with an exact title match and an optional vector
    query = {
        "vector": seed_vector,
        "filter": {
            "title": {"$eq": movie_name}
        },
        "top_k": 1,  # We only want the closest result based on metadata and vector proximity
        "include_metadata": True
    }
    
    # Perform the query on the Pinecone index
    result = index.query(**query)
    
    # Check if we have any matches and return the first one's metadata
    if result and result['matches']:
        return result['matches'][0]['metadata']
    else:
        return None
    
def get_recommendations(movie_titles, weights, num_movies):
    return None
