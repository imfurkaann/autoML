# Imports
from flask import Flask
from flask import render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
import sys
import pandas as pd
#sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autoML import autoML, trainML, analysisDATASET
import pymongo

# Init
app = Flask(__name__, template_folder="templates")

app.config["SECRET_KEY"] = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_EXTENSIONS"] = ["xlsx", "csv"]
app.config["UPLOAD_PATH"] = "datasets"

os.makedirs(app.config["UPLOAD_PATH"], exist_ok=True)

db = SQLAlchemy(app)

# Misc
def read_file(path):
    if path.split(".")[-1] == "csv":
        original_dataset = pd.read_csv(path)
    elif path.split(".")[-1] == "xlxs":
        original_dataset = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLXS.")
    return original_dataset

def connect_db(collection_name, db_name="automl_database"):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    return collection


# Handlers
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'Dosya bulunamadı'
    file = request.files['file']
    if file.filename == '':
        return 'Dosya seçilmedi'
    if file.filename.rsplit(".", 1)[1].lower() not in app.config["UPLOAD_EXTENSIONS"]:
        return 'İzin verilmeyen dosya türü'
    filepath = os.path.join(app.config['UPLOAD_PATH'], file.filename)
    file.save(filepath)
    print(filepath)
    return redirect(url_for('inputs', filepath=filepath))

@app.route('/inputs', methods=['GET', 'POST'])
def inputs():
    if request.method == "GET":
        filepath = request.args.get("filepath")
        df = read_file(filepath)
        columns = df.columns.tolist()
        return render_template('inputs.html', columns=columns, filepath=filepath) 
    else:
        filepath = request.args.get("filepath")
        target = request.form.get('target')
        test_size = float(request.form.get('test_size'))
        random_state = int(request.form.get('random_state'))
        shuffle = request.form.get('shuffle') == 'True'
        is_classification = request.form.get('is_classification') == 'True'
        metric = request.form.get('metric')
        categorical_columns = request.form.getlist('categorical_columns')
        numeric_columns = request.form.getlist('numeric_columns')
        
        ml2 = trainML(path=filepath, target=target, categorical_columns=categorical_columns, 
                 numeric_columns=numeric_columns, is_classification=is_classification, 
                 test_size=test_size, random_state=random_state, shuffle=shuffle, metric=metric)
        ml = analysisDATASET(path=filepath, target=target, categorical_columns=categorical_columns, 
                 numeric_columns=numeric_columns, is_classification=is_classification, 
                 test_size=test_size, random_state=random_state, shuffle=shuffle, metric=metric)
        ml.df_info()
        ml.before_charts()

        return redirect('result')

@app.route('/result', methods=['GET'])
def result():
    files = os.listdir("database/charts")
    png_files = sorted([file for file in files if file.endswith('.png')])
    df_info = read_file(os.path.join(app.config["UPLOAD_PATH"], "df_info.csv"))
    data = df_info.head(1000).values.tolist()
    columns = df_info.columns.tolist()
    mongocollection = connect_db("Original_Dataset")
    inputs = mongocollection.find_one()
    return render_template('result.html', charts=png_files, columns=columns, data=data, inputs=inputs)

@app.route('/start', methods=['GET'])
def start():
    datasets = os.listdir("database/datasets")
    datas = []
    for dataset in datasets:
        df = read_file(os.path.join("database", "datasets", dataset))
        data = df.values.tolist()
        columns = df.columns.tolist()
        datas.append({"name": dataset.replace("_", " ").removesuffix(".csv"), "data": data, "columns": columns})
    mongocollection = connect_db("Original_Dataset")
    inputs = mongocollection.find_one()
    return render_template('start.html', datasets=datas, inputs=inputs)

@app.route('/static/images/<path:filename>')
def serve_images(filename):
    # charts klasörünün yolunu belirtin
    directory = os.path.join(app.root_path, '..', 'database', 'charts')
    print(directory)
    # Dosyayı belirttiğimiz klasörden servise sunma
    return send_from_directory(directory, filename)
@app.route('/lightbox/<path:filename>')
def serve_lightbox(filename):
    # charts klasörünün yolunu belirtin
    directory = os.path.join(app.root_path, 'lightbox')
    print(directory)
    # Dosyayı belirttiğimiz klasörden servise sunma
    return send_from_directory(directory, filename)

### RUNNER
if __name__ == "__main__":
    app.run(debug=True)