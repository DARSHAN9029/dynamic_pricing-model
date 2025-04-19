from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)


model_path=os.path.join(os.path.dirname(__file__),'dynamic_pricing.pkl')            #loading model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)



file_path = os.path.join(os.path.dirname(__file__), 'final_data.csv')           #loading data
df = pd.read_csv(file_path)



label_encoder = LabelEncoder()
df['Title_original'] = df['Title']
df['Title'] = label_encoder.fit_transform(df['Title'])

@app.route('/')
def home():
    return "Welcome to the Dynamic Pricing API! Use POST /predict to get discount predictions."


@app.route('/predict', methods=['POST'])                    #api
def predict():
    try:
        data = request.get_json()

        product_name = data['product_name']                     # Input values
        price = float(data['price'])
        days_remaining = int(data['days_remaining'])


        if product_name in label_encoder.classes_:                                 # Encode the product name which is not in dataset
            encoded_product = label_encoder.transform([product_name])[0]
        else:
            encoded_product = -1                                            # unseen product fallback

        input_features = [[encoded_product, price, days_remaining]]

        predicted_discount = model.predict(input_features)[0]                   # Prediction
        discount_amount = (predicted_discount / 100) * price
        final_price = price - discount_amount

        result = {
            'product_name': product_name,
            'original_price': price,
            'predicted_discount_percent': round(predicted_discount, 2),
            'discount_amount': round(discount_amount, 2),
            'final_price': round(final_price, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
