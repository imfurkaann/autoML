#pandas-numpy-seaborn-matplotlib
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno

#scikit-learn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, r2_score, confusion_matrix, accuracy_score, \
     precision_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# mongodb
import pymongo


# --
import warnings
warnings.filterwarnings("ignore")
import re
<<<<<<< HEAD
import bson.binary
=======
>>>>>>> 18eb53e (app.py ve autoML.py dosyaları güncellendi)
import io
import base64
import uuid
import os
import shutil