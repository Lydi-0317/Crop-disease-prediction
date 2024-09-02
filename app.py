from flask import Flask, request, jsonify, render_template
from flask_mysqldb import MySQL
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Configure MySQL database connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Hanaph@2001'
app.config['MYSQL_DB'] = 'crop_disease_db'

mysql = MySQL(app)

# Load the trained model
model = load_model('plant_disease_model.keras')

@app.route('/')
def index():
    return render_template('index.html')


# Route for disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((128, 128))  # Resize to match the input size of your model
    img = np.array(img) / 255.0  # Normalize the image
    img = img.reshape(1, 128, 128, 3)  # Reshape for model input

    # Predict the disease
    prediction = model.predict(img)
    disease = np.argmax(prediction, axis=1)[0]

    # Fetch disease info from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM diseases WHERE id = %s", (disease,))
    disease_info = cur.fetchone()

    if disease_info is None:
        return jsonify({
        'error': 'No matching disease found',
        'disease': None,
        'precautions': None,
        'features': None
    })
    
    return jsonify({
        'disease': disease_info[1],  # Assuming the disease name is the second column
        'precautions': disease_info[2],
        'features': disease_info[3]
    })

if __name__ == '__main__':
    app.run(debug=True)
