# -*- coding: utf-8 -*-
"""Final_Year_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sBiphdjHJKDVZUfby0FScUveBPcgqORr

## Phobia Detection Using Virtual Reality/ Augmented Reality

#citations:
1. Data set:
    https://www.researchgate.net/publication/329403546_Mental_Emotional_Sentiment_Classification_with_an_EEG-based_Brain-machine_Interface

2. Research Material:
    https://www.researchgate.net/publication/335173767_A_Deep_Evolutionary_Approach_to_Bioinspired_Classifier_Optimisation_for_Brain-Machine_Interaction

3. J. J. Bird, L. J. Manso, E. P. Ribiero, A. Ekart, and D. R. Faria, “A study on mental state classification using eeg-based brain-machine     interface,”in 9th International Conference on Intelligent Systems, IEEE, 2018.

4. J. J. Bird, A. Ekart, C. D. Buckingham, and D. R. Faria, “Mental emotional sentiment classification with an eeg-based brain-machine interface,” in The International Conference on Digital Image and Signal Processing (DISP’19), Springer, 2019.

This research was part supported by the EIT Health GRaCE-AGE grant number 18429 awarded to C.D. Buckingham.

#Team Members:
1. Avishkar Borkar
2. Harshal Makote
3. Sourabh Waghmode
4. Rushikesh Sapkal

## Why this project in particular ?
## What is the scope ?

The motivation behind this project is that in today’s age, thousands of people suffer from mental hardships and conditions. I wish to raise awareness for the mental battles people have to go through and how these issues are often ignored. The project uses machine learning algorithms, statistics and complex python libraries such as scikit-learn, TensorFlow, etc. to predict if an individual has a particular phobia or not. The intention behind the project is that people have hidden fears and is an excellent idea to gradually expose them to their fears in a virtually controlled environment so that they can be worked upon and then further medical assistance can be provided.

## Approaching The Machine Learning Model

1. Data Exploration
2. Data Preparation
3. Model Evaluation
4. Confirming Model

About Dataset

<p>The data was collected from two people (1 male, 1 female) for 3 minutes per state - positive, neutral, negative. We used a Muse EEG headband which recorded the TP9, AF7, AF8 and TP10 EEG placements via dry electrodes. Six minutes of resting neutral data is also recorded, the stimuli used to evoke the emotions are below<p>

Video inputs:

1. Marley and Me - **NEGATIVE** (Twentieth Century Fox) ---> Death Scene

2. Up - **NEGATIVE** (Walt Disney Pictures) ---> Opening Death Scene

3. La La Land - **POSITIVE** (Summit Entertainment) ---> Opening musical number

4. Slow Life - **POSITIVE** (BioQuest Studios) ---> Nature timelapse

5. Funny Dogs - **POSITIVE** (MashupZone) ---> Funny dog clips

## 1. Data Exploration
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('emotions.csv')
data.head()

data.describe()

data['label'].count()

plt.bar(data['label'], height = 2500)



"""## 2. Data Preparation"""

data.isnull()

null_count = data.isnull().sum().sum()
null_count

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

missing_values_table(data)

"""### As we have 0 null values, we will proceed with the model"""

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

x = data.drop(["label"] , axis=1)

#Scaling the data
scaler = StandardScaler()
scaler.fit(x)
X = scaler.transform(x)

#Encoding data
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
Y = data['label'].copy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 42, test_size = 0.2)



"""## 3. Model Training/ Evaluation"""

final_scores = {'Support Vector Classifier' : [], 'Random Forest Classifier' : [], 'Extreme Gradient Boosting' : [], 'Logistic Regression' : []}

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

clf = svm.SVC(kernel='linear')
cv_scores = cross_val_score(clf, X, Y, cv=5)

# Evaluate the model
print("Cross-validation scores:", cv_scores)
mean_accuracy = cv_scores.mean()
final_scores['Support Vector Classifier'].append(mean_accuracy)
std_deviation = cv_scores.std()
print('--------------------------------------------------------------------')
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print('--------------------------------------------------------------------')
print(f"Standard Deviation: {std_deviation:.2f}")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
final_scores['Random Forest Classifier'].append(accuracy)
print(f'                      Accuracy: {accuracy:.2f}')
print('--------------------------------------------------------------------')
report = classification_report(y_test, y_pred)
print(report)

xgb = XGBClassifier(n_estimators = 15, max_depth = 5, learning_rate = 1)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
final_scores['Extreme Gradient Boosting'].append(accuracy)
print(f'                      Accuracy: {accuracy:.2f}')
print('--------------------------------------------------------------------')
report = classification_report(y_test, y_pred)
print(report)

lr = LogisticRegression(max_iter = 999)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
final_scores['Logistic Regression'].append(accuracy)
print(f'                      Accuracy: {accuracy:.2f}')
print('--------------------------------------------------------------------')
report = classification_report(y_test, y_pred)
print(report)



"""## 4. Evaluating All The Models"""

print(final_scores)

for key, value in final_scores.items():
    print(f'{key} : {value}%')











