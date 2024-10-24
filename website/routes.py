from .test_model import perform_segmentation
import os
from io import BytesIO
import zipfile
from flask import Flask, send_file, redirect, url_for, Blueprint, render_template, request

from flask_login import current_user

web_app = Flask(__name__)

web_app.config['UPLOAD_FOLDER'] = './data/final_nifti_files/imagesVal'
os.makedirs(web_app.config['UPLOAD_FOLDER'], exist_ok=True)

IMAGE_FOLDER = "./data/model_results"

routes = Blueprint("routes", __name__)

@routes.route("/", methods = ["GET", "POST"])
def home_page():
    return render_template("my_patients.html", account = current_user)

@routes.route('/download', methods=['GET'])
def download():
    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(IMAGE_FOLDER):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    zf.write(file_path, os.path.basename(file_path))

    memory_file.seek(0)
        
@routes.route('/results', methods = ["GET", "POST"])
def results():
    return render_template("results.html", account = current_user)

@routes.route('/uploader', methods=['POST'])
def uploader():
    if request.method == 'POST':
        individual_file = request.files['file']
        if individual_file and (individual_file.filename.endswith('.nii') or individual_file.filename.endswith('.nii.gz')):
            filepath = os.path.join(web_app.config['UPLOAD_FOLDER'], individual_file.filename)
            individual_file.save(filepath)

            primary_dir = "./data/final_nifti_files"
            model_dir = "./data/dl_models"

            test_file_name = individual_file.filename

            perform_segmentation(primary_dir, model_dir, test_file_name)

            return redirect(url_for('routes.results'))
        
        else:
            return "Please upload a valid NIFTI file (.nii or .nii.gz)."