# Dowry Prediction Flask App

This repository contains a Flask application that predicts the dowry amount based on various factors such as income, education, family size, and other demographic details. The app uses a trained machine learning model and encoded data to provide predictions.

## Features
- **User-friendly Web Interface:** Input demographic details via a form.
- **Machine Learning Integration:** Uses a pre-trained machine learning model to predict dowry amounts.
- **Data Encoding:** Encodes categorical features using label encoders to ensure consistency with the trained model.
- **Custom Predictions:** Displays the predicted dowry amount in a user-friendly format.

## Prerequisites
Ensure you have the following installed:
- Python 3.7 or above
- Flask
- Pandas
- Pickle (for loading the model and encoders)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/dowry-prediction-app.git
    cd dowry-prediction-app
    ```

2. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Add the model and encoders:**
    - Place the trained model file `dowry_model.pkl` in the `model/` directory.
    - Place the label encoders file `label_encoders.pkl` in the `model/` directory.

5. **Run the application:**
    ```bash
    python app.py
    ```

6. **Access the application:**
    Open your browser and navigate to `http://127.0.0.1:5000/`.

## File Structure
```
.
├── app.py                  # Main Flask application
├── model
│   ├── dowry_model.pkl     # Trained machine learning model
│   ├── label_encoders.pkl  # Label encoders for categorical data
├── templates
│   └── index.html          # HTML template for the web interface
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## How It Works
1. **Input Data:** Users input details like income, education, family size, location type, region, caste, religion, and ages of the bride and groom.
2. **Data Transformation:** The application encodes the categorical features using pre-trained label encoders.
3. **Prediction:** The transformed data is passed to the pre-trained model for prediction.
4. **Result Display:** The predicted dowry amount is displayed on the web interface.

## Usage
1. Launch the Flask app.
2. Fill in the form with required details.
3. Submit the form to view the predicted dowry amount.

## Requirements File
Ensure the `requirements.txt` includes the following:
```
Flask
pandas
numpy
scikit-learn
```

## Example Screenshot
Add an example screenshot of your application here.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
