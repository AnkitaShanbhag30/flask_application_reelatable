from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_recommendations

recommendations_bp = Blueprint('recommendations', __name__)

@recommendations_bp.route('/movie_recommendations', methods=['POST'])
def get_movie_recommendations():
    """
    Endpoint to retrieve movie recommendations.
    Expects a JSON payload with a list of movie titles, their respective weights, and the desired number of recommendations.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input, JSON expected'}), 400
    
    movie_titles = data.get('movie_titles')
    weights = data.get('weights', [])  # Weights are optional
    num_movies = data.get('num_movies', 10)  # Default to 10 movies if not provided
    
    if not movie_titles or not isinstance(movie_titles, list):
        return jsonify({'error': 'Movie titles list is required'}), 400
    if weights and not all(isinstance(w, (int, float)) for w in weights):
        return jsonify({'error': 'Weights must be a list of numbers'}), 400
    if not isinstance(num_movies, int) or num_movies <= 0:
        return jsonify({'error': 'num_movies must be a positive integer'}), 400
    
    try:
        recommendations = get_recommendations(movie_titles, weights, num_movies)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
