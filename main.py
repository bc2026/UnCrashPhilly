import geopy
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Function to get the zipcode from latitude and longitude
def get_zipcode(lat, lon, geolocator):
    location = geolocator.reverse((lat, lon), exactly_one=True)
    return location.raw['address']['postcode']

# Set up the geolocator
geolocator = geopy.Nominatim(user_agent='my-insurance-application')

# Load data
df = pd.read_csv('data.csv')
df['Zipcode'] = df.apply(lambda row: get_zipcode(row['X'], row['Y'], geolocator), axis=1)

# Clean data
df = df.drop(columns=['X', 'Y']).dropna().drop_duplicates()

# Prepare data for the model
X = pd.get_dummies(df['Zipcode'])
y = df['automobile']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train the model
model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.fit(X_train, y_train, epochs=10, batch_size=12)

# Predict and evaluate the model
y_pred = model.predict(X_test)
y_pred = [0 if val < 0.5 else 1 for val in y_pred]
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save the model
model.save('insurance_rate_model.h5')

# Retrieve live traffic data from MapQuest API

bounding_box = '40.0094,-75.13333'  # Center 
api_key = 'lsqO1n1Ejdy02BvlDAHcun0OPK7U2SiM'of Philadelphia
url = f'https://www.mapquestapi.com/traffic/v2/incidents?key={api_key}&boundingBox={bounding_box}&filters=incidents'

response = requests.get(url)
if response.status_code == 200:
    traffic_data = response.json()
    print(traffic_data)
else:
    print("Failed to retrieve traffic data:", response.status_code)

# Output the dataframe as an HTML table
df.to_html('index.html', index=False)
