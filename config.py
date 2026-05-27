import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'database', 'tumor_detection.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = 'brain_tumor_secret_key_2024'

# Upload folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'static', 'reports')

# Model
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')

# Allowed extensions
ALLOWED_EXTENSIONS = {'gz', 'nii'}