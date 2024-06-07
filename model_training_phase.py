import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def train_model(emotions_csv):
    data = pd.read_csv(emotions_csv)

    scaler = StandardScaler()
    scaler.fit(data.drop("label", axis=1))  # Scale features without dropping names
    X = scaler.transform(data.drop("label", axis=1)) 

    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    Y = data['label'].copy() 

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 42, test_size = 0.2)

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    with open('saved_model.pkl', 'wb') as f:  # Open in binary write mode
        pickle.dump(clf, f)