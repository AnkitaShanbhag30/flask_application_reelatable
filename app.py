from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/check', methods=['GET'])
def check_endpoint():
    # Respond with a JSON message
    return jsonify({"message": "Endpoint functional"})

@app.route('/api', methods=['POST'])
def lookup_movie():
    data = request.json  # Get JSON data sent from Flutter app
    movie_name = data.get('movieName', '')
    print(f"Looking up the movie - name: {movie_name}")
    # Respond with a JSON message
    return jsonify({"message": "Looking up the movie"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
