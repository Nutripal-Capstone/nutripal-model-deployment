from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd

app = Flask(__name__)


model = tf.keras.models.load_model('food_model.h5')

X_train_mean = pd.read_csv('mean.csv', index_col=0)
X_train_std = pd.read_csv('std.csv', index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0)
meal_types = pd.read_csv('meal_types.csv', index_col=0)
if len(X_train_mean.columns) == 1:
    X_train_mean = X_train_mean[X_train_mean.columns[0]]

if len(X_train_std.columns) == 1:
    X_train_std = X_train_std[X_train_std.columns[0]]

if len(meal_types.columns) == 1:
    meal_types = meal_types[meal_types.columns[0]]

@app.route('/', methods=['GET'])
def home():
    return "API is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)  

    calorie = data['calorie']  

    # Normalize user input
    user_input = pd.DataFrame([[calorie]], columns=['calorie_normalized'])
    user_input_norm = (user_input - X_train_mean) / X_train_std

    # Get meal recommendations
    recommendations = model.predict(user_input_norm)

    response = []

    for i, recommendation in enumerate(recommendations[0]):
        recommended_foods = y_train[(y_train['meal_type'] == meal_types[i])][['food_id', 'serving_id', 'food_name', 'calories']]
        recommended_food_samples = recommended_foods.sample(n=10)
        for index, row in recommended_food_samples.iterrows():
            food_id = row['food_id']
            serving_id = row['serving_id']
            food_name = row['food_name']
            calories = row['calories']
            response.append({
                'food_id': food_id,
                'serving_id': serving_id,
                'food_name': food_name,
                'calories': calories,
            })

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)