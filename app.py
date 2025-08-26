import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Sidebar - model selection
st.sidebar.title("ML Playground")
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("Decision Tree", "Random Forest", "Logistic Regression", "SVM", "KNN")
)

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Moons", "Circles", "Blobs", "Classification (3D)")
)

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)

# Function to load dataset
def get_dataset(name):
    if name == "Moons":
        return make_moons(n_samples=500, noise=0.3, random_state=42)
    elif name == "Circles":
        return make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42)
    elif name == "Blobs":
        return make_blobs(n_samples=500, centers=2, random_state=42)
    else:  # 3D dataset
        return make_classification(n_samples=500, n_features=3, n_classes=2,
                                   n_informative=3, n_redundant=0, random_state=42)

X, y = get_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model selection
def get_classifier(name):
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == "Logistic Regression":
        return LogisticRegression()
    elif name == "SVM":
        return SVC(kernel="rbf", probability=True)
    else:
        return KNeighborsClassifier(n_neighbors=5)

model = get_classifier(classifier_name)

# Train model
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

st.write(f"### Model: {classifier_name}")
st.write(f"Accuracy: **{acc:.2f}**")

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)
st.write(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# GridSearchCV (only for small demo, keep params light)
if st.sidebar.checkbox("Run GridSearchCV"):
    params = {}
    if classifier_name == "Decision Tree":
        params = {"max_depth": [3, 5, 7, None]}
    elif classifier_name == "Random Forest":
        params = {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
    elif classifier_name == "SVM":
        params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    elif classifier_name == "KNN":
        params = {"n_neighbors": [3, 5, 7]}
    
    grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
    grid.fit(X, y)
    st.write("Best Params:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)

# Save/Load Model
col1, col2 = st.columns(2)
with col1:
    if st.button("Export Model"):
        with open("saved_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Model exported as saved_model.pkl")

with col2:
    if st.button("Import Model"):
        try:
            with open("saved_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            st.success("Model imported successfully")
            st.write("Imported Model Accuracy:", loaded_model.score(X_test, y_test))
        except:
            st.error("No saved_model.pkl found!")

# Plot Decision Boundary (2D)
if X.shape[1] == 2:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    st.pyplot(plt.gcf())

# Plot Decision Boundary (3D)
if X.shape[1] == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50)
    st.pyplot(fig)
