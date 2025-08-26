# Decision-Tree-Playground-decision-Boundary-Tree-Graph-Accuracy-

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
