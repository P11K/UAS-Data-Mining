from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('gbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = [
        request.form.get('marital_status'),
        float(request.form.get('application_mode')),
        float(request.form.get('application_order')),
        float(request.form.get('course')),
        float(request.form.get('daytime_evening_attendance')),
        float(request.form.get('previous_qualification')),
        float(request.form.get('previous_qualification_grade')),
        float(request.form.get('nationality')),
        float(request.form.get('mothers_qualification')),
        float(request.form.get('fathers_qualification')),
        float(request.form.get('mothers_occupation')),
        float(request.form.get('fathers_occupation')),
        float(request.form.get('admission_grade')),
        float(request.form.get('displaced')),
        float(request.form.get('educational_special_needs')),
        float(request.form.get('debtor')),
        float(request.form.get('tuition_fees_up_to_date')),
        request.form.get('gender'),
        float(request.form.get('scholarship_holder')),
        float(request.form.get('age_at_enrollment')),
        float(request.form.get('international')),
        float(request.form.get('curricular_units_1st_sem_credited')),
        float(request.form.get('curricular_units_1st_sem_enrolled')),
        float(request.form.get('curricular_units_1st_sem_evaluations')),
        float(request.form.get('curricular_units_1st_sem_approved')),
        float(request.form.get('curricular_units_1st_sem_grade')),
        float(request.form.get('curricular_units_1st_sem_without_evaluations')),
        float(request.form.get('curricular_units_2nd_sem_credited')),
        float(request.form.get('curricular_units_2nd_sem_enrolled')),
        float(request.form.get('curricular_units_2nd_sem_evaluations')),
        float(request.form.get('curricular_units_2nd_sem_approved')),
        float(request.form.get('curricular_units_2nd_sem_grade')),
        float(request.form.get('curricular_units_2nd_sem_without_evaluations')),
        float(request.form.get('unemployment_rate')),
        float(request.form.get('inflation_rate')),
        float(request.form.get('gdp'))
    ]
    
    # Transform categorical features if needed
    # For simplicity, assuming Gender is binary (0 or 1), adjust as necessary
    features[17] = 1 if features[17].lower() == 'male' else 0  # Assuming 'male' and 'female' are the possible values
    
    # Combine features into a numpy array
    features = np.array(features).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make a prediction
    prediction = model.predict(scaled_features)
    
    # Map prediction to output
    output = 'Success' if prediction == 1 else 'Dropout'  # Adjust based on your actual labels
    
    return render_template('index.html', prediction_text=f'Student will {output}')

if __name__ == "__main__":
    app.run(debug=True)
