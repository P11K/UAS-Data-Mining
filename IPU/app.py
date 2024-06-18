from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the scaler and models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('svm_model_category.pkl', 'rb') as f:
    model_category = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    model_critical = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    pm10 = float(request.form.get('pm10'))
    pm25 = float(request.form.get('pm25'))
    so2 = float(request.form.get('so2'))
    co = float(request.form.get('co'))
    o3 = float(request.form.get('o3'))
    no2 = float(request.form.get('no2'))
    max_value = float(request.form.get('max'))

    # Store features in a list
    features = [pm10, pm25, so2, co, o3, no2, max_value]

    # Find the actual maximum value among the inputs
    actual_max_value = max(pm10, pm25, so2, co, o3, no2)

    # Check if the max value matches the actual maximum value and identify the matching variable
    if max_value == actual_max_value:
        if max_value == pm10:
            matching_variable = "PM10"
        elif max_value == pm25:
            matching_variable = "PM25"
        elif max_value == so2:
            matching_variable = "SO2"
        elif max_value == co:
            matching_variable = "CO"
        elif max_value == o3:
            matching_variable = "O3"
        elif max_value == no2:
            matching_variable = "NO2"
    else:
        matching_variable = "Unknown"

    # Combine features into a numpy array
    features = np.array(features).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make predictions
    prediction_category = model_category.predict(scaled_features)
    prediction_critical = model_critical.predict(scaled_features)

    category_labels = ['SEDANG', 'TIDAK SEHAT']  # Adjust based on actual labels


    if prediction_category.size > 0 and prediction_critical.size > 0:
        # Adjust labels based on predictions
        if prediction_category[0] == 1:  # Assuming 'SEDANG'
            output_category = category_labels[0]
        elif prediction_category[0] == 2:  # Assuming 'TIDAK SEHAT'
            output_category = category_labels[1]
        
        output_critical = matching_variable if max_value == actual_max_value else "Unknown"
    else:
        output_category = "Unknown"
        output_critical = "Unknown"

    return render_template('index.html', prediction_text=f'Category: {output_category}, Critical: {output_critical}')

if __name__ == "__main__":
    app.run(debug=True)
