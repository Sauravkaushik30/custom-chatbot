from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    ...
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

