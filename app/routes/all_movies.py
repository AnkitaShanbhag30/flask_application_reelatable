from flask import Blueprint, request, jsonify
from app.services.pinecone_service import get_all_movies

all_movies_bp = Blueprint('all_movies', __name__)

@all_movies_bp.route('/get_all_movies', methods=['GET'])
def all_movies():
    movie_data = get_all_movies()
    if movie_data:
        return jsonify(movie_data)
    else:
        return jsonify({'error': 'Movies not found'}), 404