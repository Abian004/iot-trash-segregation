from flask import Flask, request, jsonify, render_template
import os
from roboflow import Roboflow
from google.cloud import storage
import uuid
import logging
import requests
import tempfile

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Roboflow client
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
project = rf.workspace().project("smart-segregation-bin-sxfgl")
model = project.version(6).model

# Cloud Storage setup for uploaded images
bucket_name = os.environ.get("GCP_BUCKET_NAME")
print("Using bucket:", bucket_name)
storage_client = storage.Client()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test-upload', methods=['POST'])
def test_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    filename = image_file.filename
    
    # Just return confirmation that we received the file
    return jsonify({
        'success': True,
        'filename': filename,
        'size': len(image_file.read())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        
        # Save image to Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_file(image_file)
        
        # Generate a signed URL (valid for 5 minutes)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=300,
            method="GET"
        )
        
        # Download the image from the signed URL to a temporary local file
        response = requests.get(signed_url)
        if response.status_code != 200:
            raise Exception("Failed to download image from signed URL")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Get prediction using the Roboflow SDK on the local file
        predictions = model.predict(temp_path, confidence=40, overlap=30).json()
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Return prediction results
        return jsonify(predictions)
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This is used when running locally
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
