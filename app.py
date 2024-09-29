from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the trained model
with open('model/dowry_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    income = form_data['income']
    education = form_data['education']
    family_size = int(form_data['family_size'])
    location_type = form_data['location_type']
    region = form_data['region']
    caste = form_data['caste']
    religion = form_data['religion']
    age_bride = int(form_data['age_bride'])
    age_groom = int(form_data['age_groom'])

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'income': [income],
        'education': [education],
        'family_size': [family_size],
        'location_type': [location_type],
        'region': [region],
        'caste': [caste],
        'religion': [religion],
        'age_bride': [age_bride],
        'age_groom': [age_groom]
    })

    # Apply same encoding as during training
    input_data['income'] = label_encoders['income'].transform(input_data['income'])
    input_data['education'] = label_encoders['education'].transform(input_data['education'])
    input_data['location_type'] = label_encoders['location_type'].transform(input_data['location_type'])
    input_data['region'] = label_encoders['region'].transform(input_data['region'])
    input_data['caste'] = label_encoders['caste'].transform(input_data['caste'])
    input_data['religion'] = label_encoders['religion'].transform(input_data['religion'])

    # Predict dowry amount
    prediction = model.predict(input_data)
    return render_template('index.html', prediction_text=f'Predicted Dowry Amount: ₹{int(prediction[0]):,}')

if __name__ == "__main__":
    app.run(debug=True)