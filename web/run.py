# Imports
from flask import Flask
from flask import render_template, request, redirect, url_for, session, send_from_directory
import os
import sys
import pandas as pd
#sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autoML import trainML, analysisDATASET
import pymongo
import itertools

# Init
app = Flask(__name__, template_folder="templates")

app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_EXTENSIONS"] = ["xlsx", "csv"]
app.config["UPLOAD_PATH"] = "datasets"

os.makedirs(app.config["UPLOAD_PATH"], exist_ok=True)

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
    client = pymongo.MongoClient("mongodb://mongo:27017/")
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

        session["filepath"] = filepath
        session["target"] = target
        session["test_size"] = test_size
        session["random_state"] = random_state
        session["shuffle"] = shuffle
        session["is_classification"] = is_classification
        session["metric"] = metric
        session["categorical_columns"] = categorical_columns
        session["numeric_columns"] = numeric_columns
        
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
    files = os.listdir("database/charts/Original_Dataset")
    png_files = sorted([file for file in files if file.endswith('.png')])
    df_info = read_file(os.path.join(app.config["UPLOAD_PATH"], "df_info.csv"))
    data = df_info.head(1000).values.tolist()
    columns = df_info.columns.tolist()
    mongocollection = connect_db("Original_Dataset")
    inputs = mongocollection.find_one()
    return render_template('result.html', charts=png_files, columns=columns, data=data, inputs=inputs)

@app.route('/start', methods=['GET'])
def start():
    mongocollection = connect_db("Original_Dataset")
    inputs = mongocollection.find_one()
    ml = trainML(path=inputs["path"], target=inputs["target"], categorical_columns=inputs["categorical_columns"], 
                 numeric_columns=inputs["numeric_columns"], is_classification=inputs["is_classification"], 
                 test_size=inputs["test_size"], random_state=inputs["random_state"], shuffle=inputs["shuffle"], metric=session.get("metric"))
    dummy_df = ml.dummy_df()
    missing_data = ml.missing_data_fill(dummy_df)
    outlier_data = ml.outliers(list(missing_data))
    df_list = ml.all_datasets(missing_data, outlier_data)
    after_charts = ml.after_charts(df_list)
    
    datasets = os.listdir("database/datasets")
    datas = []
    for dataset in datasets:
        df = read_file(os.path.join("database", "datasets", dataset))
        data = df.values.tolist()
        columns = df.columns.tolist()
        try:
            files = os.listdir("database/charts/"+dataset.removesuffix(".csv"))
        except FileNotFoundError:
            continue
        charts = sorted([file for file in files if file.endswith(".png")])
        datas.append({"dataset": dataset.removesuffix(".csv"), "name": dataset.replace("_", " ").removesuffix(".csv"), "data": data, "columns": columns, "charts": charts})

    return render_template('start.html', datasets=datas, inputs=inputs)

@app.route("/train", methods=["GET"])
def train():
    mongocollection = connect_db("Original_Dataset")
    inputs = mongocollection.find_one()
    ml = trainML(path=inputs["path"], target=inputs["target"], categorical_columns=inputs["categorical_columns"], 
                 numeric_columns=inputs["numeric_columns"], is_classification=inputs["is_classification"], 
                 test_size=inputs["test_size"], random_state=inputs["random_state"], shuffle=inputs["shuffle"], metric=session["metric"])

    client = pymongo.MongoClient("mongodb://mongo:27017/")  # MongoDB bağlantı URL'nizi girin
    db = client["automl_database"]  # Veritabanı adınızı girin
    # path değerlerini toplayacağımız liste
    path_listesi = []
    # Tüm koleksiyonları al
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        # Koleksiyondaki tüm dökümanlarda path alanını çek
        for doc in collection.find({}, {"path": 1}):  # Sadece path alanını alıyoruz
            path_value = doc.get("path")
            if path_value:  # Eğer path değeri varsa listeye ekle
                path_listesi.append(path_value)

    df_list = []
    for df_path in path_listesi:
        df = read_file(df_path)
        df.name = df_path.rsplit("/", 1)[1].rsplit(".", 1)[0].replace("_", " ")
        if df.name == "original dataset" or df.name == "original dataset2":
            continue
        df_list.append(df)

    train_alg = ml.train_alg(df_list)
    metrics = ml.metrics(train_alg)
    best_model = ml.best_model(metrics)
    best_model.sort(key= lambda x: x[0])
    grouped = itertools.groupby(best_model, key=lambda x: x[0])
    return render_template("train.html", inputs=inputs, best_models=best_model, grouped=grouped)


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