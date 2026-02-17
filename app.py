from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Feature order MUST match training dataset
        features = [
            float(request.form['age']),        # age
            int(request.form['sex']),          # sex
            int(request.form['cp']),           # chest pain
            float(request.form['trestbps']),   # resting BP
            float(request.form['chol']),       # cholesterol
            int(request.form['fbs']),          # fasting blood sugar
            int(request.form['restecg']),      # rest ECG
            float(request.form['thalach']),    # max heart rate
            int(request.form['exang']),        # exercise angina
            float(request.form['oldpeak']),    # ST depression
            int(request.form['slope']),        # slope
            int(request.form['ca']),           # major vessels
            int(request.form['thal'])          # thal
        ]

        # Convert to numpy array
        final_input = np.array([features])

        # Prediction
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]*100

        # Result formatting
        if prediction == 1:
            result = f"⚠ High Risk of Heart Disease ({probability:.2f}% confidence)"
        else:
            result = f"✅ Low Risk of Heart Disease ({100 - probability:.2f}% confidence)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
