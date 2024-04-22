from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from app.routes.metadata import metadata_bp
from app.routes.recommendations import recommendations_bp
from app.routes.patterns import patterns_bp
from app.services.pinecone_service import search_movies

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(metadata_bp, url_prefix='/metadata')
app.register_blueprint(recommendations_bp, url_prefix='/recommendations')
app.register_blueprint(patterns_bp, url_prefix='/patterns')

@app.route('/search', methods=['POST'])
def hybrid_search():
    try:
        data = request.json
        traits = data.get('traits')
        top_k = data.get('top_k', 10)
        alpha = data.get('alpha', 0.5)
        if not traits or not isinstance(traits, list):
            raise ValueError("Invalid or missing 'traits' parameter; it must be a list.")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("'top_k' must be a positive integer.")
        if not (0 <= alpha <= 1):
            raise ValueError("'alpha' must be a float between 0 and 1.")

        results = search_movies(traits, top_k, alpha)
        return jsonify(results)
    except ValueError as ve:
        abort(400, description=str(ve))
    except Exception as e:
        abort(500, description=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)