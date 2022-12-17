import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay


def read_dataset(filename: str, encoding: str = 'utf-8', separator=',') -> pd.DataFrame:
    df = pd.read_csv(filename, encoding=encoding, sep=separator)
    print('Dataset leido.')
    return df


def encode_labels(df: pd.DataFrame, col_index: int) -> pd.DataFrame:
    le = preprocessing.LabelEncoder()
    df.values[:, col_index] = le.fit(df.values[:, col_index])
    return df


def split_dataset(df: pd.DataFrame) -> tuple:
    X = df.values[:, 3:]
    y = df.values[:, 1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=100)
    print('Dataset dividido.')
    return X_train, X_test, y_train, y_test


def predict(X_test, model):
    y_pred = model.predict(X_test)
    print('Predicción finalizada.')
    return y_pred


def calculate_accuracy(y_test, y_pred) -> tuple:
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy calculado.')
    return confusion, accuracy, report


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# LECTURA DEL DATASET

df = read_dataset('api/data_analysis/respuestas-cleaned.csv', 'latin-1', '|')
df = encode_labels(df, 1)

# SEPARACIÓN DE DATOS DE ENTRENAMIENTO Y PRUEBA

X_train, X_test, y_train, y_test = split_dataset(df)


def execute_algorithms():
    for i in range(len(classifiers)):
        print(f'\n--------------------\n{names[i]}\n--------------------')
        classifiers[i].fit(X_train, y_train)
        print(f'Modelo entrenado.')
        y_pred = predict(X_test, classifiers[i])
        confusion, accuracy, report = calculate_accuracy(y_test, y_pred)
        print(f'Confusion matrix:\n {confusion}')
        print(f'Accuracy score: {accuracy}')
        print(f'Classification report:\n {report}')
        print()
