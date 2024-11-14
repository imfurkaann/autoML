from imports import *


class autoML():

<<<<<<< HEAD
    "aaaaaaaaaaaaaaaaaaaaaaaaaaa"

=======
>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
    def __init__(self, path, target, categorical_columns, numeric_columns, is_classification, test_size, random_state, shuffle, metric):
        self.path = path
        self.target = target
        self.is_classification = is_classification
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.metric = metric

        self.original_dataset = self.read_file(self.path, self.target)
        self.original_dataset.name = "Original Dataset"

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

        self.database_dir, self.datasets_dir, self.charts_dir = self.database_files()

    def connect_db(self, collection_name, db_name="automl_database"):
         
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        return self.collection

    def read_file(self, path, target):

        if path.split(".")[-1] == "csv":
            original_dataset = pd.read_csv(path)
        elif path.split(".")[-1] == "xlxs":
            original_dataset = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLXS.")
        
        original_dataset[target] = original_dataset[target].astype('category')

        return original_dataset

    def database_files(self):

        database_dir = os.path.join("database")
        datasets_dir = os.path.join(database_dir, "datasets")
        charts_dir = os.path.join(database_dir, "charts")
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)

        new_file_path = os.path.join(datasets_dir, "original_dataset.csv")
        shutil.copy(self.path, new_file_path)

        return database_dir, datasets_dir, charts_dir
    
    def columns(self):

        if self.categorical_columns is None:
            self.categorical_columns =[column for column in self.original_dataset.columns if is_categorical_dtype(self.original_dataset[column]) and column != self.target]
        if self.numeric_columns is None:
            self.numeric_columns = [column for column in self.original_dataset.columns if is_numeric_dtype(self.original_dataset[column])]
        
        # Düzeltilecek database kaydetme kısmı
        return self.categorical_columns, self.numeric_columns
    
    
class analysisDATASET(autoML):
<<<<<<< HEAD
=======
    
    

>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def df_info(self):

        df_unique = pd.DataFrame(self.original_dataset.nunique()).rename(columns={0:"Unique"})
        df_missing = pd.DataFrame(self.original_dataset.isnull().sum()).rename(columns={0:"Missing Values"})
        df_stats = pd.DataFrame(self.original_dataset.describe().T)
        df_info = pd.concat([df_unique, df_missing, df_stats], axis = 1)
<<<<<<< HEAD

        return df_info
    
    def before_charts(self):
        
        charts_list = []

=======
        
        df_info.to_csv("df_info.csv", index=True)

        return df_info
    

    def before_charts(self):
        new_dir = os.path.join(self.charts_dir, self.original_dataset.name.replace(" ", "_"))
        os.makedirs(new_dir, exist_ok=True)

        charts_list = []

        # Update database to ensure charts array exists
>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
        self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one(
            {"dataset_name": self.original_dataset.name},
            {"$setOnInsert": {"charts": []}},
            upsert=True
        )
<<<<<<< HEAD
    

        #target dist chart
        target_dist_fig, target_dist_ax = plt.subplots()
        sns.countplot(x=self.target, data=self.original_dataset, ax=target_dist_ax)
        target_dist_ax.set_title(f'{self.target} Değişkeninin Dağılımı')
        target_dist_fig.savefig(f"{self.charts_dir}\\Hedef_Değişken_Dağılımı.png")
        self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$push":{"charts":f"{self.charts_dir}\\Hedef_Değişken_Dağılımı.png"}})
        charts_list.append(target_dist_fig)

        #columns dist chart
        for col in self.original_dataset.columns:
            if self.original_dataset[col].dtype != 'object':  
                columns_dist_fig, columns_dist_ax = plt.subplots(figsize=(10,6))
                sns.histplot(data=self.original_dataset[col], kde=True, ax=columns_dist_ax)
                columns_dist_ax.set_title(f'{col} Değişkeninin Dağılımı')
                
                columns_dist_fig.savefig(f"{self.charts_dir}\\{col}_Degiskeni_Dagilimi.png")
                self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$push":{"charts":f"{self.charts_dir}\\{col}_Degiskeni_Dagilimi.png"}})
                charts_list.append(columns_dist_fig)
        
        #corr matrix
        corr_fig, corr_ax = plt.subplots(figsize = (10,8))
        sns.heatmap(data=self.original_dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=corr_ax)
        corr_ax.set_title("Korelasyon Matrisi")
        corr_fig.savefig(f"{self.charts_dir}\\Korelasyon_Matrisi.png")
        self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$push":{"charts":f"{self.charts_dir}\\Korelasyon_Matrisi.png"}})
        charts_list.append(corr_fig)
        
        #outlier charts
        for col in self.original_dataset.columns:
            if self.original_dataset[col].dtype != 'object':  
                
                outlier_fig, outlier_ax = plt.subplots(figsize = (10,8))
                sns.boxplot(x=self.original_dataset[col], ax=outlier_ax)
                outlier_ax.set_title(f'{col} Değişkeninde Aykırı Değerler')
                outlier_fig.savefig(f"{self.charts_dir}\\{col}_Aykırı_Değerler.png")
                self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$push":{"charts":f"{self.charts_dir}\\{col}_Aykırı_Değerler.png"}})
                charts_list.append(outlier_fig)

        #missing data
        missing_fig, missing_ax = plt.subplots(figsize=(10, 6)) 
        msno.bar(self.original_dataset, ax=missing_ax)  
        missing_ax.set_title('Eksik Verilerin Bar Grafiği')
        missing_fig.savefig(f"{self.charts_dir}\\Eskik_Veri_Bar_Grafiği.png")
        self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$push":{"charts":f"{self.charts_dir}\\{col}_Eskik_Veri_Bar_Grafiği.png"}})
        charts_list.append(missing_fig)

        return charts_list
=======

        # Target distribution chart
        target_dist_fig, target_dist_ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=self.target, data=self.original_dataset, ax=target_dist_ax)
        target_dist_ax.set_title(f'{self.target} Değişkeninin Dağılımı')
        target_dist_fig.savefig(f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Hedef_Değişken_Dağılımı.png")
        self.connect_db(self.original_dataset.name.replace(' ', '_')).update_one({"dataset_name": self.original_dataset.name}, 
                                                                                {"$push": {"charts": f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Hedef_Değişken_Dağılımı.png"}})
        charts_list.append(target_dist_fig)
        plt.close(target_dist_fig)  # Close to free memory

        # Columns distribution charts
        for col in self.original_dataset.columns:
            if self.original_dataset[col].dtype != 'object':
                columns_dist_fig, columns_dist_ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=self.original_dataset[col], kde=True, ax=columns_dist_ax)
                columns_dist_ax.set_title(f'{col} Değişkeninin Dağılımı')

                columns_dist_fig.savefig(f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\{col}_Degiskeni_Dagilimi.png")
                self.connect_db(self.original_dataset.name.replace(' ', '_')).update_one({"dataset_name": self.original_dataset.name}, 
                                                                                        {"$push": {"charts": f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\{col}_Degiskeni_Dagilimi.png"}})
                charts_list.append(columns_dist_fig)
                plt.close(columns_dist_fig)  # Close to free memory

        # Correlation matrix
        corr_fig, corr_ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data=self.original_dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=corr_ax)
        corr_ax.set_title("Korelasyon Matrisi")
        corr_fig.savefig(f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Korelasyon_Matrisi.png")
        self.connect_db(self.original_dataset.name.replace(' ', '_')).update_one({"dataset_name": self.original_dataset.name}, 
                                                                                {"$push": {"charts": f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Korelasyon_Matrisi.png"}})
        charts_list.append(corr_fig)
        plt.close(corr_fig)  # Close to free memory

        # Outlier charts
        for col in self.original_dataset.columns:
            if self.original_dataset[col].dtype != 'object':
                outlier_fig, outlier_ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=self.original_dataset[col], ax=outlier_ax)
                outlier_ax.set_title(f'{col} Değişkeninde Aykırı Değerler')
                outlier_fig.savefig(f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\{col}_Aykırı_Değerler.png")
                self.connect_db(self.original_dataset.name.replace(' ', '_')).update_one({"dataset_name": self.original_dataset.name}, 
                                                                                        {"$push": {"charts": f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\{col}_Aykırı_Değerler.png"}})
                charts_list.append(outlier_fig)
                plt.close(outlier_fig)  # Close to free memory

        # Missing data chart
        missing_fig, missing_ax = plt.subplots(figsize=(8, 6)) 
        msno.bar(self.original_dataset, ax=missing_ax)
        missing_ax.set_title('Eksik Verilerin Bar Grafiği')
        missing_fig.savefig(f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Eskik_Veri_Bar_Grafiği.png")
        self.connect_db(self.original_dataset.name.replace(' ', '_')).update_one({"dataset_name": self.original_dataset.name}, 
                                                                                {"$push": {"charts": f"{self.charts_dir}\\{self.original_dataset.name.replace(' ', '_')}\\Eskik_Veri_Bar_Grafiği.png"}})
        charts_list.append(missing_fig)
        plt.close(missing_fig)  # Close to free memory

        return charts_list

>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
        

class trainML(autoML):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.alg_list = self.alg_list(self.is_classification)
        self.database_files()

        existing_datasets = self.connect_db(self.original_dataset.name.replace(" ", "_")).distinct("dataset_name")
<<<<<<< HEAD

        if self.original_dataset.name not in existing_datasets:  
                db_dataset = self.original_dataset.to_dict("records")
                data = {
                    "dataset_name": self.original_dataset.name, "path": f"{self.datasets_dir}\\original_dataset.csv", 
                    "target": self.target, "test_size": self.test_size,
                    "random_state": self.random_state, "shuffle": self.shuffle, "is_classification": self.is_classification, 
                    "categorical_columns": self.categorical_columns, "numeric_columns": self.numeric_columns
                }
                self.connect_db(self.original_dataset.name.replace(" ", "_")).insert_one(data)
=======
        
        if self.original_dataset.name not in existing_datasets:
            db_dataset = self.original_dataset.to_dict("records")
            data = {
                "dataset_name": self.original_dataset.name, 
                "path": f"{self.datasets_dir}\\original_dataset.csv", 
                "target": self.target, 
                "test_size": self.test_size,
                "random_state": self.random_state, 
                "shuffle": self.shuffle, 
                "is_classification": self.is_classification, 
                "categorical_columns": self.categorical_columns, 
                "numeric_columns": self.numeric_columns
            }
            
            self.connect_db(self.original_dataset.name.replace(" ", "_")).insert_one(data)
        else:
            data = {
                "dataset_name": self.original_dataset.name, 
                "path": f"{self.datasets_dir}\\original_dataset.csv", 
                "target": self.target, 
                "test_size": self.test_size,
                "random_state": self.random_state, 
                "shuffle": self.shuffle, 
                "is_classification": self.is_classification, 
                "categorical_columns": self.categorical_columns, 
                "numeric_columns": self.numeric_columns
            }
            
            result = self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one(
                {"dataset_name": self.original_dataset.name},  
                {"$set": data},  
                upsert=True  
            )

>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
    
    def dummy_df(self):

        df_dummy = pd.get_dummies(self.original_dataset, columns=self.categorical_columns)
        df_dummy.to_csv(f"{self.datasets_dir}\\df_original_dummy.csv", index=False)

        self.connect_db(self.original_dataset.name.replace(" ", "_")).update_one({"dataset_name":self.original_dataset.name}, 
                                                                        {"$set":{"dataset_dummy_path":f"{self.datasets_dir}\\df_original_dummy.csv"}})

        return df_dummy
    
    def missing_data_fill(self, dummy_dataset):

        knn_imputer = KNNImputer(n_neighbors=4)
        scaler = MinMaxScaler()
        
        df_imputed = dummy_dataset.copy()
        df_scaler = pd.DataFrame(scaler.fit_transform(df_imputed), columns = df_imputed.columns)
        df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_scaler), columns=df_scaler.columns)
        df_imputed.name = "KNN ile Doldurulmuş Dataset"

        df_drop = dummy_dataset.copy()
        df_drop = df_drop.dropna()
        df_drop.name = "Eksik Veriler Atılmış Dataset"

        df_mean = dummy_dataset.copy()
        df_mean.name = "Ortalama ile Doldurulmuş Dataset"

        df_median = dummy_dataset.copy()
        df_median.name = "Medyan ile Doldurulmuş Dataset"

        for column in self.numeric_columns:

            df_mean[column] = df_mean[column].fillna(df_mean[column].mean())
            df_median[column] = df_median[column].fillna(df_median[column].median())
        
        return df_mean, df_median, df_imputed, df_drop
    
    def outliers(self, datasets):

        outliers_list = []

        for df in datasets:

            df_outlier_deleted = df.copy() 
            df_outlier_winsorized = df.copy() 

            for column in self.numeric_columns:

                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                df_outlier_deleted = df_outlier_deleted[~((df_outlier_deleted[column] < (Q1 - 1.5 * IQR)) | (df_outlier_deleted[column] > (Q3 + 1.5 * IQR)))]

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_outlier_winsorized[column] = df_outlier_winsorized[column].clip(lower=lower_bound, upper=upper_bound)

            df_outlier_winsorized.name = df.name + " ve Winsorized ile Aykırı Değerler Düzeltilmiş"
            df_outlier_deleted.name = df.name + " ve Aykırı Değerler Silinmiş"

            outliers_list.append(df_outlier_winsorized)
            outliers_list.append(df_outlier_deleted)

        return outliers_list  

    def all_datasets(self, missing_df, outlier_list):

        df_list = list(missing_df) + list(outlier_list)
        

        for df in df_list:
            
            df.to_csv(f"{self.datasets_dir}\\{df.name.replace(" ", "_")}.csv", index=False)
            existing_datasets = self.connect_db(df.name.replace(" ", "_")).distinct("dataset_name")

            if df.name not in existing_datasets:  

                data = {
                    "dataset_name": df.name, "path": f"{self.datasets_dir}\\{df.name.replace(" ", "_")}.csv", "target": self.target, "test_size": self.test_size,
                    "random_state": self.random_state, "shuffle": self.shuffle, "is_classification": self.is_classification, 
                    "categorical_columns": self.categorical_columns, "numeric_columns": self.numeric_columns
                }
                self.connect_db(df.name.replace(" ", "_")).insert_one(data)

        return df_list

    def after_charts(self, dataset_list):
<<<<<<< HEAD
    
        for df in dataset_list:

            #target dist chart
            target_dist_fig, target_dist_ax = plt.subplots()
            sns.countplot(x=self.target, data=df, ax=target_dist_ax)
            target_dist_ax.set_title(f'{self.target} Değişkeninin Dağılımı')

            #columns dist chart
            dist_charts = []
            for col in df.columns:
                if df[col].dtype != 'object':  
                    columns_dist_fig, columns_dist_ax = plt.subplots(figsize=(10,6))
                    sns.histplot(data=df[col], kde=True, ax=columns_dist_ax)
                    columns_dist_ax.set_title(f'{col} Değişkeninin Dağılımı')
                    dist_charts.append(columns_dist_fig)
            
            #corr matrix
            corr_fig, corr_ax = plt.subplots(figsize = (10,8))
            sns.heatmap(data=df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=corr_ax)
            corr_ax.set_title("Korelasyon Matrisi")
            
            #outlier charts
            outlier_charts = []
            for col in df.columns:
                if df[col].dtype != 'object':  
                    
                    outlier_fig, outlier_ax = plt.subplots(figsize = (10,8))
                    sns.boxplot(x=df[col], ax=outlier_ax)
                    outlier_ax.set_title(f'{col} Değişkeninde Aykırı Değerler')
                    outlier_charts.append(outlier_fig)

            #missing data
            missing_fig, missing_ax = plt.subplots(figsize=(10, 6)) 
            msno.bar(df, ax=missing_ax)  
            missing_ax.set_title('Eksik Verilerin Bar Grafiği') 

            with open('path/to/smallimage.jpg', 'rb') as image_file:
                encoded_image = bson.binary.Binary(image_file.read())
                self.db.insert_one({"charts": "smallimage.jpg", "data": encoded_image})

        return target_dist_fig, dist_charts, corr_fig, outlier_charts, missing_fig
=======
        for df in dataset_list:
            new_dir = os.path.join(self.charts_dir, df.name.replace(" ", "_"))
            os.makedirs(new_dir, exist_ok=True)

            charts_list = []

            # Update database to ensure charts array exists
            self.connect_db(df.name.replace(" ", "_")).update_one(
                {"dataset_name": df.name},
                {"$setOnInsert": {"charts": []}},
                upsert=True
            )

            # Target distribution chart
            target_dist_fig, target_dist_ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=self.target, data=df, ax=target_dist_ax)
            target_dist_ax.set_title(f'{self.target} Değişkeninin Dağılımı')
            target_dist_fig.savefig(f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Hedef_Değişken_Dağılımı.png")
            self.connect_db(df.name.replace(" ", "_")).update_one({"dataset_name":df.name},
                                                                    {"$push":{"charts":f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Hedef_Değişken_Dağılımı.png"}})
            charts_list.append(target_dist_fig)
            plt.close(target_dist_fig)  # Close to free memory

            # Columns distribution charts
            for col in df.columns:
                if df[col].dtype != 'object':
                    columns_dist_fig, columns_dist_ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(data=df[col], kde=True, ax=columns_dist_ax)
                    columns_dist_ax.set_title(f'{col} Değişkeninin Dağılımı')
                    columns_dist_fig.savefig(f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\{col}_Degiskeni_Dagilimi.png")
                    self.connect_db(df.name.replace(" ", "_")).update_one({"dataset_name":df.name},
                                                                            {"$push":{"charts":f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\{col}_Degiskeni_Dagilimi.png"}})
                    charts_list.append(columns_dist_fig)
                    plt.close(columns_dist_fig)  # Close to free memory

            # Correlation matrix
            corr_fig, corr_ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data=df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=corr_ax)
            corr_ax.set_title("Korelasyon Matrisi")
            corr_fig.savefig(f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Korelasyon_Matrisi.png")
            self.connect_db(df.name.replace(" ", "_")).update_one({"dataset_name":df.name},
                                                                    {"$push":{"charts":f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Korelasyon_Matrisi.png"}})
            charts_list.append(corr_fig)
            plt.close(corr_fig)  # Close to free memory

            # Outlier charts
            for col in df.columns:
                if df[col].dtype != 'object':
                    outlier_fig, outlier_ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=df[col], ax=outlier_ax)
                    outlier_ax.set_title(f'{col} Değişkeninde Aykırı Değerler')
                    outlier_fig.savefig(f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\{col}_Aykırı_Değerler.png")
                    self.connect_db(df.name.replace(" ", "_")).update_one({"dataset_name":df.name},
                                                                            {"$push":{"charts":f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\{col}_Aykırı_Değerler.png"}})
                    charts_list.append(outlier_fig)
                    plt.close(outlier_fig)  # Close to free memory

            # Missing data chart
            missing_fig, missing_ax = plt.subplots(figsize=(8, 5)) 
            msno.bar(df, ax=missing_ax)
            missing_ax.set_title('Eksik Verilerin Bar Grafiği')
            missing_fig.savefig(f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Eskik_Veri_Bar_Grafiği.png")
            self.connect_db(df.name.replace(" ", "_")).update_one({"dataset_name":df.name},
                                                                    {"$push":{"charts":f"{self.charts_dir}\\{df.name.replace(' ', '_')}\\Eskik_Veri_Bar_Grafiği.png"}})
            charts_list.append(missing_fig)
            plt.close(missing_fig)  # Close to free memory
>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)


    def train_test_splitt(self, df):

        X = df.drop(self.target, axis=1)  
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)

        return X_train, X_test, y_train, y_test 
    
    def alg_list(self, is_classification):

        if self.is_classification:
            
            modelXGB = XGBClassifier()
            modelLOGISTIC = LogisticRegression()
            modelLGB = LGBMClassifier()
            modelRFOREST = RandomForestClassifier()
            modelSVC = svm.SVC(kernel='linear')

            model_list = [modelXGB, modelLOGISTIC, modelLGB, modelRFOREST, modelSVC]

        else:

            modelXGB = XGBClassifier()
            modelLINEAR = LinearRegression()
            modelLGB = LGBMClassifier()
            modelRFOREST = RandomForestClassifier()
            modelSVC = svm.SVC(kernel='linear')

            model_list = [modelXGB, modelLINEAR, modelLGB, modelRFOREST, modelSVC]

        return model_list
    
    def train_alg(self, df_list):
        
        model_dict = {}

        for df in df_list:  
                      
            model_dict[df.name] = {
                'Models': [],
                'Dataset': df
            }
            
            train_data = self.train_test_splitt(df)

            for alg in self.alg_list:
                model = alg
                model.fit(train_data[0], train_data[2])
                model_dict[df.name]['Models'].append(model)

        return model_dict
    
    def metrics(self, models):

        model_metrics = {}
        for key, value in models.items():

            model_list = value['Models']  
            data = value['Dataset']  
            model_metrics[key] = {}

            traindata = self.train_test_splitt(data)

            for idx, model in enumerate(model_list):
                
                
                y_pred = model.predict(traindata[1])  

                if self.is_classification:
                    metrics = {
                        "Accuracy": accuracy_score(traindata[3], y_pred),
                        "ROC-AUC Score": roc_auc_score(traindata[3], y_pred),
                        "Precision Score": precision_score(traindata[3], y_pred),
                        "Recall Score": recall_score(traindata[3], y_pred)
                    }
                else:
                    metrics = {
                        'MAE': mean_absolute_error(traindata[3], y_pred),
                        'RMSE': np.sqrt(mean_squared_error(traindata[3], y_pred)),
                        "R2 Score": r2_score(traindata[3], y_pred),
                    }
                
                self.connect_db(key.replace(" ", "_")).update_one({"dataset_name":key}, {"$set":{"metrics":metrics}})
                model_metrics[key][str(model)[:-2]] = metrics

            model_names = [model.__class__.__name__ for model in model_list]
            self.connect_db(key.replace(" ", "_")).update_one({"dataset_name":key}, {"$set":{"trained_models":model_names}})


        return model_metrics
    
    def best_model(self, model_metrics):

        best_model_list = []

        for datasets, models in model_metrics.items():
            
            for model, metrics in models.items():

                best_model_list.append((datasets, model, metrics[self.metric]))

        best_model = sorted(best_model_list, key=lambda x: x[2], reverse=True)

        return best_model