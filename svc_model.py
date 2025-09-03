from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

def train_svc(X_train, y_train):
    svc = LinearSVC(max_iter=1000)
    svc.fit(X_train, y_train)
    return svc

def evaluate_svc(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - Support Vector Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
def tune_svc(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],     
        'max_iter': [1000, 2000]
    }
    svc = LinearSVC(max_iter=1000)
    grid = GridSearchCV(svc, param_grid, scoring='accuracy', cv=5, verbose = 1, n_jobs = -1)
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_

def plot_roc_svc(X_train, y_train, X_test, y_test, label_encoder):
    svc = LinearSVC(max_iter=1000)
    calibrated_model = CalibratedClassifierCV(svc)
    calibrated_model.fit(X_train, y_train)
    
    y_proba = calibrated_model.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

    plt.figure(figsize=(10, 8))
    for i in range(y_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Support Vector Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()