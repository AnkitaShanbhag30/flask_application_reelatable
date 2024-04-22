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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)