from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_representative_traits

patterns_bp = Blueprint('patterns', __name__)

@patterns_bp.route('/get_movie_patterns', methods=['POST'])
def get_movie_patterns():
    data = request.get_json()
    if not data or 'titles' not in data:
        return jsonify({'error': 'No movie titles provided in request body'}), 400

    movie_titles = data['titles']
    if not isinstance(movie_titles, list):
        return jsonify({'error': 'Titles must be provided as a list'}), 400

    representative_traits = get_representative_traits(movie_titles)
    if representative_traits:
        return jsonify(representative_traits)
    else:
        return jsonify({'error': 'Unable to find or process movie traits'}), 404