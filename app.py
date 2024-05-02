from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from xgboost import XGBClassifier
import numpy as np
import pickle
with open("xgb.pkl",'rb') as f:
    xgb = pickle.load(f)
print(xgb)
app = Flask(__name__)
CORS(app)
gender_mapping = {'male': 0, 'female': 1}
country_mapping = {'france': 0, 'spain': 1, 'germany': 2}
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON array from the request
    json_data = request.json
    
    json_data['gender'] = gender_mapping[json_data["gender"]]
    json_data['country'] = country_mapping[json_data["country"]]

    columns_to_convert = ['gender','credit_score', 'country', 'age', 'balance', 'tenure', 'products_number', 'credit_card', 'active_member','estimated_salary']

    # Convert values to integers
    for key in columns_to_convert:
        if isinstance(json_data[key], str) or isinstance(json_data[key], np.int64):
            json_data[key] = int(json_data[key])
     

    pred=xgb.predict([list(json_data.values())])
    print(f"This is pred:{pred}")
    return jsonify({'prediction':int(pred[0])})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
