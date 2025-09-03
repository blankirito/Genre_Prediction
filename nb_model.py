from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_nb(X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb

def evaluate_nb(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    labels_in_test = sorted(list(set(y_test)))
    target_names = [label_encoder.classes_[i] for i in labels_in_test]
    print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def tune_nb(X_train, y_train):
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
    nb = MultinomialNB()
    grid = GridSearchCV(nb, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best Parameters for Naive Bayes:", grid.best_params_)
    return grid.best_estimator_

def plot_roc_nb(model, X_test, y_test, label_encoder):
    y_proba = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

    plt.figure(figsize=(10, 8))
    for i in range(y_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Multinomial NB')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
