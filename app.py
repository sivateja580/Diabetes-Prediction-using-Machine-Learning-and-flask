from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("classifier_model.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from the form
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])

    # Scale the input data if required
    scaled_data = sc.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]
    result = "🩸 Diabetic" if prediction == 1 else "✅ Not Diabetic"

    # Return result page
    return render_template("result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
