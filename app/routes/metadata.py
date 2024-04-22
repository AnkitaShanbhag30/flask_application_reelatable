from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_metadata_by_movie, search_movies

metadata_bp = Blueprint('metadata', __name__)

@metadata_bp.route('/movie_metadata', methods=['GET'])
def movie_metadata():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400

    metadata = get_metadata_by_movie(title)
    if metadata:
        return jsonify(metadata)
    else:
        return jsonify({'error': 'Movie not found'}), 404

@metadata_bp.route('/search_by_traits', methods=['POST'])
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
        results = search_movies(traits, num_results)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500