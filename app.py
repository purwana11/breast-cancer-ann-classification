from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'breast_cancer_prediction_secret_key'

# Load model dan scaler
model = load_model("ann_breast_cancer_model.h5")
scaler = joblib.load("scaler_breast_cancer.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil semua input fitur dari form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    result = model.predict(features_scaled)
    probability = float(result[0][0])
    
    if probability > 0.5:
        prediction = 'Malignant (Ganas)'
        status = 'danger'
        recommendation = 'Segera konsultasikan dengan dokter spesialis onkologi untuk pemeriksaan lebih lanjut dan penanganan yang tepat.'
    else:
        prediction = 'Benign (Jinak)'
        status = 'success'
        recommendation = 'Tetap lakukan pemeriksaan rutin sesuai anjuran dokter untuk memantau kondisi kesehatan Anda.'
    
    # Simpan hasil ke session untuk ditampilkan di result.html
    session['prediction'] = prediction
    session['probability'] = probability
    session['status'] = status
    session['recommendation'] = recommendation
    
    return redirect(url_for('result'))

@app.route('/result')
def result():
    prediction = session.get('prediction', 'Tidak ada hasil')
    probability = session.get('probability', 0)
    status = session.get('status', 'info')
    recommendation = session.get('recommendation', '')
    
    return render_template('result.html', 
                         prediction=prediction, 
                         probability=probability,
                         status=status,
                         recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
