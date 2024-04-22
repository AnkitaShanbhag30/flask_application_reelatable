import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
from app.utils.helpers import load_pca_model
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

def get_sparse_embeddings(traits_dict, n_components=12):
    """Generates a single sparse embedding for a dictionary of trait lists."""
    category_embeddings = []
    pca = load_pca_model('app/utils/pca_model.pkl')
    # Calculate embeddings for each category and average them
    for key in keys_to_process:
        traits_list = traits_dict.get(key, [])
        if traits_list:
            embeddings = embeddings_model.encode(traits_list)
            category_avg_embedding = np.mean(embeddings, axis=0)
            category_embeddings.append(category_avg_embedding)
    
    # Check if there are any category embeddings to process
    if category_embeddings:
        # Average across category embeddings to get a final single embedding
        final_embedding = np.mean(category_embeddings, axis=0)

        # Apply PCA if dimensionality reduction is needed
        if len(final_embedding) > n_components:
            reduced_embedding = pca.transform(final_embedding.reshape(1, -1)).flatten()
        else:
            reduced_embedding = final_embedding

        # Convert to Pinecone format or similar
        indices = list(range(len(reduced_embedding)))
        values = reduced_embedding.tolist()
        final_pinecone_embedding = {"indices": indices, "values": values}

        return final_pinecone_embedding
    else:
        return None  # or handle empty input case appropriately


def get_dense_embeddings(texts):
    """
    Generates a single dense embedding vector that represents the average embedding of the given texts.
    
    :param texts: A list of text strings.
    :return: A single averaged dense embedding vector.
    """
    # Encode all texts into embeddings, each cleaned by removing newlines
    embeddings = embeddings_model.encode([t.replace("\n", " ") for t in texts])

    # Calculate the mean of these embeddings along axis 0 to get a single embedding
    if embeddings.size == 0:
        # Handle the case where there are no texts or embeddings cannot be generated
        return np.zeros(embeddings_model.get_sentence_embedding_dimension())
    else:
        averaged_embedding = np.mean(embeddings, axis=0)

    return averaged_embedding

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

def hybrid_query(vector, sparse_vector, top_k, filter=None, include_metadata=True):
    """
    Query Pinecone with both dense and sparse vector embeddings.
    :param vector: Dense vector for the query, which is a list of numpy arrays
    :param sparse_vector: Sparse vector for the query
    :param top_k: Number of top results to return
    :param include_metadata: Whether to include metadata in the query results
    :return: Search results from Pinecone
    """
    # Convert each numpy array in the vector list to a Python list
    if any(isinstance(elem, np.ndarray) for elem in vector):
        vector = [elem.tolist() if isinstance(elem, np.ndarray) else elem for elem in vector]

    # Ensure sparse vector values are lists if they're numpy arrays
    if 'values' in sparse_vector and isinstance(sparse_vector['values'], np.ndarray):
        sparse_vector['values'] = sparse_vector['values'].tolist()

    if filter :
        query = {
            "vector": vector,
            "sparse_vector": sparse_vector,
            "filter": filter,
            "top_k": top_k,
            "include_metadata": include_metadata
        }
    else :
        query = {
            "vector": vector,
            "sparse_vector": sparse_vector,
            "top_k": top_k,
            "include_metadata": include_metadata
        }        
    return index.query(**query)


def search_movies(traits_dict, texts, filter=None, top_k=10, alpha=0.5):
    """
    Search movies by given traits and text descriptions using hybrid search.
    """
    sparse_embeddings = get_sparse_embeddings(traits_dict)
    print("Sparse Embeddings done")

    dense_embeddings = get_dense_embeddings(texts)
    print("Dense Embeddings done")

    dense_vector, sparse_vector = hybrid_scale(dense_embeddings, sparse_embeddings, alpha)
    print("Scaling Embeddings done")

    results = hybrid_query(dense_vector, sparse_vector, top_k, filter)
    print("Hybrid index query done")

    return [match['metadata'] for match in results['matches']]


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
    
def extract_traits(metadata, keys_to_process):
    traits_dict = {key: [] for key in keys_to_process}
    for key in keys_to_process:
        i = 1
        while True:
            trait_key = f"{key}_{i}_trait"
            if trait_key in metadata:
                traits_dict[key].append(metadata[trait_key])
                i += 1
            else:
                break
    return traits_dict

def get_recommendations(movie_titles, num_movies, alpha):
    """
    Generate recommendations based on a list of movie titles using hybrid search with both dense and sparse embeddings.
    """
    metadata_list = [get_metadata_by_movie(title) for title in movie_titles]
    plotlines = [metadata['overview'] for metadata in metadata_list if metadata]
    traits_dicts = [extract_traits(metadata, keys_to_process) for metadata in metadata_list if metadata]

    # Combining all traits from all movies into a single dictionary for sparse embeddings
    combined_traits = {}
    for key in keys_to_process:
        combined_traits[key] = [trait for traits_dict in traits_dicts for trait in traits_dict.get(key, [])]

    query_filter = {"title": {"$nin": movie_titles}}

    # Perform the search using the combined traits and plotlines
    results = search_movies(combined_traits, plotlines, filter=query_filter, top_k=num_movies, alpha=alpha)

    return results
