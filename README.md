# Decision-Tree-Playground-decision-Boundary-Tree-Graph-Accuracy-
<img width="686" height="683" alt="image" src="https://github.com/user-attachments/assets/8f3e2b95-2a56-4e3b-9d9d-c327cf62f650" />

# README.md


- **Datasets**: Use built-in datasets (Iris, Wine, Breast Cancer) or upload your own CSV.
- **Hyperparameters**: Control `criterion`, `splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, `ccp_alpha`.
- **Training & Evaluation**:
- Train/test split
- Accuracy score
- Classification report
- Confusion matrix
- ROC curve (for binary classification)
- **Visuals**:
- Decision boundary (choose 2 features)
- Full decision tree graph
- Feature importances


---


## 📦 Installation


Clone the repository and install requirements:


```bash
git clone https://github.com/yourusername/decision-tree-playground.git
cd decision-tree-playground
pip install -r requirements.txt
```


---


## ▶️ Run the App


```bash
streamlit run app.py
```


Then open the link Streamlit provides (usually `http://localhost:8501`).


---


## 📊 Usage


1. Select a dataset from the sidebar or upload your own CSV.
2. If uploading a CSV, specify the **target column**.
3. Adjust hyperparameters (criterion, splitter, max_depth, etc.).
4. Click **🚀 Run / Train Model** to see results.


---
# Decision Boundary ML Playground 🎯

An interactive Streamlit app to explore classifiers like Decision Tree, Random Forest, Logistic Regression, SVM, and KNN.

## 🚀 Features
- Multiple datasets (moons, circles, blobs, synthetic 3D).
- Model selection and training.
- Accuracy + Cross-validation results.
- Hyperparameter tuning (GridSearchCV).
- Export/Import trained models with pickle.
- Visualize decision boundaries (2D + 3D).

## 🛠 Installation
```bash
pip install -r requirements.txt


## 🛠 Requirements
- Python 3.8+
- Libraries (see `requirements.txt`):
- streamlit
- scikit-learn
- matplotlib
- pandas
- numpy


---


## 📸 Screenshots


- Decision boundary plots
- Confusion matrix
- Tree graph with splits


---
