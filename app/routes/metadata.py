from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_metadata_by_movie, search_movies

metadata_bp = Blueprint('metadata', __name__)

@metadata_bp.route('/get_movie_metadata', methods=['GET'])
def movie_metadata():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400

    metadata = get_metadata_by_movie(title)
    if metadata:
        return jsonify(metadata)
    else:
        return jsonify({'error': 'Movie not found'}), 404