from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_recommendations, search_movies

recommendations_bp = Blueprint('recommendations', __name__)

@recommendations_bp.route('/get_movie_recommendations', methods=['POST'])
def get_movie_recommendations():
    """
    Endpoint to retrieve movie recommendations based on movie titles and their traits.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input, JSON expected'}), 400
    
    movie_titles = data.get('movie_titles')
    alpha = data.get('alpha', 0.5)  # Default alpha if not provided
    num_movies = data.get('num_movies', 10)  # Default to 10 movies if not specified
    
    if not movie_titles or not isinstance(movie_titles, list):
        return jsonify({'error': 'Movie titles list is required'}), 400
    if not isinstance(num_movies, int) or num_movies <= 0:
        return jsonify({'error': 'num_movies must be a positive integer'}), 400
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        return jsonify({'error': 'alpha must be a number between 0 and 1'}), 400
    
    try:
        recommendations = get_recommendations(movie_titles, num_movies, alpha)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@recommendations_bp.route('/search_by_traits', methods=['POST'])
def search_by_traits():
    """
    Endpoint to search movies by various traits.
    Expects a JSON with a dictionary containing traits.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input, JSON expected'}), 400

    traits = data.get('traits')
    num_results = data.get('num_results', 10)
    
    if not traits:
        return jsonify({'error': 'Traits are required for the search'}), 400

    # Search movies by traits
    try:
        results = search_movies(traits, texts=[""], filter=None, top_k=num_results, alpha=0.0) # alpha=0.0 ensures that only sparse embeddings of traits are used for this search
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
