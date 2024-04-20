from flask import Flask, request, render_template
import geopy
import pandas as pd
from keras.models import load_model

app = Flask(__name__)
geolocator = geopy.Nominatim(user_agent='my-geo-app')
model = load_model('insurance_rate_model.h5')  # Load your pre-trained model

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Render the home page with the form

@app.route('/predict', methods=['POST'])
def predict():
    neighborhood = request.form['neighborhood']
    location = geolocator.geocode(neighborhood)
    if location:
        zipcode = geolocator.reverse((location.latitude, location.longitude), exactly_one=True).raw['address']['postcode']
        prediction = model.predict(pd.get_dummies(pd.Series([zipcode])))
        return render_template('result.html', prediction=prediction[0])
    else:
        return 'Could not geocode the inputted neighborhood.'

if __name__ == '__main__':
    app.run(debug=True)
