# Dependencies
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import pickle
from flask_cors import CORS

# Your API definition
app = Flask(__name__)
CORS(app)


# The function should return None or ended without a return statement.
@app.route('/', methods=['POST'])
def predict():
    if lr:
        try:
            request.headers = {'Content-Type': 'application/json'}
            json_ = request.json
            df = pd.DataFrame.from_dict(json_, orient="index").T
            prediction = lr.predict(df)[0]

            return jsonify({'prediction': str(prediction)})
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    with open("back-end/model.plk", "rb") as file:
        lr = pickle.load(file)
        print(lr)
    app.run(debug=True)
