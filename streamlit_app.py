
import io
import textwrap
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ----------------------------
# Utility
# ----------------------------

def load_builtin(name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        raise ValueError("Unknown dataset")

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    classes = list(data.target_names)
    return X, y, classes


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric columns; drop others for modeling simplicity
    num_df = df.select_dtypes(include=[np.number]).copy()
    return num_df


def plot_decision_regions(model, X2: np.ndarray, y: np.ndarray, feature_names: List[str], class_names: List[str]):
    # X2 must be (n_samples, 2)
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Contourf for regions
    cs = ax.contourf(xx, yy, Z, alpha=0.25)
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k", s=30)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Decision Boundary (selected 2 features)")
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')

    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_tree_graph(clf: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names if class_names else True,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        ax=ax,
    )
    ax.set_title("Decision Tree")
    fig.tight_layout()
    return fig


def plot_feature_importances(clf: DecisionTreeClassifier, feature_names: List[str]):
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig


# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="Decision Tree Classifier Playground", layout="wide")
st.title("🌳 Decision Tree Classifier – Visual Playground")

with st.sidebar:
    st.header("1) Data")
    data_mode = st.radio("Choose data source", ["Built-in", "Upload CSV"], horizontal=True)

    if data_mode == "Built-in":
        dataset_name = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer"], index=0)
        X, y, class_names = load_builtin(dataset_name)
        target_col = "target"
    else:
        uploaded = st.file_uploader("Upload CSV (includes features and target)", type=["csv"]) 
        target_col = st.text_input("Target column name (required)")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())
            if target_col and target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
                X = ensure_numeric(X)
                class_names = sorted([str(c) for c in pd.unique(y)])
            else:
                X = y = class_names = None
        else:
            X = y = class_names = None

    st.markdown("---")
    st.header("2) Model Hyperparameters")
    criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0,
                             help="Split quality measure")
    splitter = st.selectbox("splitter", ["best", "random"], index=0,
                            help="How to choose split at each node")
    max_depth = st.slider("max_depth (None=unlimited)", 1, 30, value=5)
    max_depth = None if st.checkbox("Use unlimited depth", value=False) else max_depth
    min_samples_split = st.slider("min_samples_split", 2, 50, value=2)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 50, value=1)
    min_impurity_decrease = st.number_input("min_impurity_decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.0001,
                                           help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value")
    ccp_alpha = st.number_input("ccp_alpha (Minimal Cost-Complexity Pruning)", min_value=0.0, max_value=1.0, value=0.0, step=0.0001)

    st.markdown("---")
    st.header("3) Train/Test")
    test_size = st.slider("test_size", 0.1, 0.5, value=0.2, step=0.05)
    scale_features = st.checkbox("Standardize features", value=False,
                                 help="Helpful for decision boundary visualization; not required for trees")
    random_state = st.number_input("random_state (seed)", min_value=0, max_value=9999, value=42, step=1)


# Validate data
if (data_mode == "Built-in") or (data_mode == "Upload CSV" and uploaded is not None and X is not None and y is not None):
    # Feature selection for 2D boundary
    all_features = list(X.columns)
    st.subheader("Choose features for decision boundary (2D)")
    col_a, col_b = st.columns(2)
    with col_a:
        f1 = st.selectbox("Feature 1", all_features, index=0)
    with col_b:
        # default to different feature if possible
        default_idx = 1 if len(all_features) > 1 else 0
        f2 = st.selectbox("Feature 2", all_features, index=default_idx)

    # Prepare data
    X_model = X.copy()

    # Optional scaling – applies to all features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_model[:] = scaler.fit_transform(X_model.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=test_size, random_state=random_state, stratify=y if len(pd.unique(y)) > 1 else None
    )

    # Build model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )

    # Train
    run_clicked = st.button("🚀 Run / Train Model", type="primary")
    if run_clicked:
        clf.fit(X_train, y_train)

        # Predict & metrics
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy on test set: **{acc:.4f}**")

        # Classification report text
        report = classification_report(y_test, y_pred, target_names=class_names if class_names else None)
        with st.expander("Classification Report"):
            st.code(report, language="text")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = plot_confusion_matrix(cm, class_names=class_names if class_names else [str(i) for i in np.unique(y)])
        st.pyplot(cm_fig, use_container_width=True)

        # Decision boundary (2D)
        if f1 != f2 and len(all_features) >= 2:
            # For boundary we fit a separate tree on just the 2 features for a clean 2D projection
            X2_train = X_train[[f1, f2]].values
            X2_test = X_test[[f1, f2]].values
            clf2 = DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                random_state=random_state,
            )
            clf2.fit(X2_train, y_train)
            boundary_fig = plot_decision_regions(clf2, np.vstack([X2_train, X2_test]), np.hstack([y_train, y_test]), [f1, f2], class_names)
            st.pyplot(boundary_fig, use_container_width=True)
        else:
            st.info("Select two different features to draw a 2D decision boundary.")

        # ROC (binary only)
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            try:
                if hasattr(clf, "predict_proba"):
                    y_score = clf.predict_proba(X_test)[:, 1]
                else:
                    # Fall back to decision function if available
                    y_score = clf.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=unique_classes[1])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve (binary)")
                ax.legend(loc="lower right")
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"ROC could not be computed: {e}")

        # Tree graph
        tree_fig = plot_tree_graph(clf, feature_names=list(X_model.columns), class_names=class_names)
        st.pyplot(tree_fig, use_container_width=True)

        # Feature importances
        fi_fig = plot_feature_importances(clf, list(X_model.columns))
        if fi_fig is not None:
            st.pyplot(fi_fig, use_container_width=True)

else:
    st.info("Upload a CSV and specify the target column, or use a built-in dataset from the sidebar.")

# Footer tip
st.caption("Tip: Use 'min_impurity_decrease' and 'ccp_alpha' to control tree growth and reduce overfitting. Toggle standardization for smoother decision regions.")
