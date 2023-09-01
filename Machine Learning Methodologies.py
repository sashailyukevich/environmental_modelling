import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv("file_path")
'''
DATA EXPLORATION with visual tools:
'''
    # 1
plt.figure()
df.boxplot(by="variety", figsize=(15,10))
plt.show
    # 2
plt.figure()
sns.pairplot(df, hue="variety", height=3, markers=["o", "s", "D"])
plt.show
    # 3
plt.figure()
sns.lmplot(data=df, x='feature_1', y='feature_2', hue="variety", height=10, scatter_kws={"s":100})
plt.show   
    # 4
fig, ax = plt.subplots()
fig.set_size_inches(12,9)
sns.swarmplot(data=df, x='variety', y='feature', hue='variety', size=6)
plt.show
    # 5
from pandas.plotting import parallel_coordinates
fig, ax = plt.subplots()
fig.set_size_inches(12,9)
parallel_coordinates(df, "variety", color=['blue', 'red', 'green'])

'''
SUPERVISED MACHINE LEARNING

    CLASSIFIERS
        Step 1 - Get Data
        Step 2 - Explore and Clean Data
        Step 3 - Prepare Data: Feature and Target
        Step 4 - Train Model
        Step 5 - Run Model
'''
FEATURE_1 = input()
FEATURE_2 = input()
FEATURE_3 = input()
FEATURE_4 = input()

TARGET = input()

FEATURE_1_FOR_PREDICTION = float(input())
FEATURE_2_FOR_PREDICTION = float(input())
FEATURE_3_FOR_PREDICTION= float(input())
FEATURE_4_FOR_PREDICTION = float(input())

'''
KNN
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

N_NEIGHBORS = int(input())

knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
CV = int(input())

def find_best_k_parameter(df: pd.DataFrame) -> plt:

    feature_df = df[[FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4]]
    target_df = df[[TARGET]]
    
    X = feature_df
    y = target_df
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=22)

    k_range = range(1,26)
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, np.ravel(y_train))
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
    
    plt.plot(k_range, scores)
    plt.xlabel("k values for KNN")
    plt.ylabel("Accuracy scores")
    plot = plt.show()

    return plot

def train_knn_model(df: pd.DataFrame) -> float:

    feature_df = df[[FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4]]
    target_df = df[[TARGET]]
    
    X = feature_df
    y = target_df

    cross_validation = cross_validate(knn, X, y, cv=CV)
    mean_score = cross_validation['test_score'].mean()

    print(mean_score)

# OR

def train_knn_model_with_for_loop(df: pd.DataFrame) -> list:

    feature_df = df[[FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4]]
    target_df = df[[TARGET]]
    
    X = feature_df
    y = target_df

    for i in range(2,20):
        cross_validation = cross_validate(knn, X, y, cv=i)
        mean_score = cross_validation['test_score'].mean()
        
        print(mean_score)

def run_knn_model(df: pd.DataFrame) -> pd.array:

    feature_df = df[[FEATURE_1, FEATURE_2, FEATURE_3, FEATURE_4]]
    target_df = df[[TARGET]]
    
    X = feature_df
    y = target_df

    knn.fit(X,y)
    knn.predict([[
        FEATURE_1_FOR_PREDICTION,
        FEATURE_2_FOR_PREDICTION,
        FEATURE_3_FOR_PREDICTION,
        FEATURE_4_FOR_PREDICTION,
        ]])
    
    return knn.predict
