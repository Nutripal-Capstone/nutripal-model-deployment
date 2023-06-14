from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load model kalian disini seperti pada kasus ini menggunakan model linear
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET'])
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Lakukan Preprocess data terlebih dahulu jika ada
    data_predict = int(data["data"])
    print(data_predict)

    # Make predictions using the loaded model
    predictions = model.predict([[1000]])

    # Lakukan Postprocess data terlebih dahulu jika ada
    # ...
    print(predictions[0])
    # Return the predictions as a JSON response
    # return (predictions)
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
