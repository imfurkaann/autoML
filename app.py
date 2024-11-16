from autoML import autoML, trainML, analysisDATASET

# Genel Tanımlamalar
path = "datasets/diabestes2.csv"   
target = "Outcome"
test_size = 0.6
random_state = 58
shuffle = True
is_classification = True
metric = "Accuracy"
categorical_columns = ["Pregnancies"]
numeric_columns = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]


ml = trainML(path=path, target=target, categorical_columns=categorical_columns, 
                 numeric_columns=numeric_columns, is_classification=is_classification, 
                 test_size=test_size, random_state=random_state, shuffle=shuffle, metric=metric)

a = analysisDATASET(path=path, target=target, categorical_columns=categorical_columns, 
                 numeric_columns=numeric_columns, is_classification=is_classification, 
                 test_size=test_size, random_state=random_state, shuffle=shuffle, metric=metric)

# ml.df_info()
a.before_charts()



dummy_df = ml.dummy_df()
missing_data = ml.missing_data_fill(dummy_df)
outlier_data = ml.outliers(list(missing_data))
df_list = ml.all_datasets(missing_data, outlier_data)
after_charts = ml.after_charts(df_list)

# train_alg = ml.train_alg(df_list)
# metrics = ml.metrics(train_alg)
# best_model = ml.best_model(metrics)    