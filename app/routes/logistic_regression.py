# Logistic Regression Route

from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/api/logistic-regression/fit', methods=['POST'])
def fit():
    data = request.json
    X = data['X']  # Features
    y = data['y']  # Target
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X, y)
    return jsonify({'status': 'success'}), 201