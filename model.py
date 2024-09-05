import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from flask import Flask, request, jsonify

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

model = LinearRegression().fit(X, y)

joblib.dump(model, 'model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array([data['input']]).reshape(-1, 1))
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=8080)
