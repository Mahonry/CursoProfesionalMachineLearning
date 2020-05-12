import joblib
import numpy as np 

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods = ['GET'])
def predict():
    X_test = np.array([7.59,7.48,1.62,1.53,0.8,0.64,0.36,0.32,2.28])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion': list(prediction)})


if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)