from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model/heart_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get numeric inputs
    Age = float(request.form['Age'])
    Sex = int(request.form['Sex'])           # 1=Male, 0=Female
    RestingBP = float(request.form['RestingBP'])
    Cholesterol = float(request.form['Cholesterol'])
    FastingBS = float(request.form['FastingBS'])
    MaxHR = float(request.form['MaxHR'])
    ExerciseAngina = int(request.form['ExerciseAngina'])  # 1=Yes,0=No
    Oldpeak = float(request.form['Oldpeak'])

    # Get categorical inputs and one-hot encode manually
    chest = request.form['ChestPainType']       # ATA, NAP, TA
    restecg = request.form['RestingECG']        # Normal, ST
    stslope = request.form['ST_Slope']          # Flat, Up

    # ChestPainType
    chest_ATA = 1 if chest == 'ATA' else 0
    chest_NAP = 1 if chest == 'NAP' else 0
    chest_TA  = 1 if chest == 'TA' else 0

    # RestingECG
    restecg_Normal = 1 if restecg == 'Normal' else 0
    restecg_ST     = 1 if restecg == 'ST' else 0

    # ST_Slope
    stslope_Flat = 1 if stslope == 'Flat' else 0
    stslope_Up   = 1 if stslope == 'Up' else 0

    # Combine all features in same order as model
    final_input = np.array([Age, Sex, RestingBP, Cholesterol, FastingBS, MaxHR, ExerciseAngina, Oldpeak,
                            chest_ATA, chest_NAP, chest_TA,
                            restecg_Normal, restecg_ST,
                            stslope_Flat, stslope_Up]).reshape(1, -1)

    # Predict
    prediction = model.predict(final_input)[0]
    result = "Patient may have heart disease." if prediction == 1 else "Patient is likely healthy."

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
