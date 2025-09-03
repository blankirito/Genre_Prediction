from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_pac(X_train, y_train):
    pac = PassiveAggressiveClassifier()
    pac.fit(X_train, y_train)
    return pac

def evaluate_pac(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    labels_in_test = sorted(list(set(y_test)))
    target_names = [label_encoder.classes_[i] for i in labels_in_test]
    print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - PassiveAggressiveClassifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def tune_pac(X_train, y_train):
    param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],             
    'max_iter': [500, 1000, 1500],          
    'loss': ['hinge', 'squared_hinge']      
    }
    
    pac = PassiveAggressiveClassifier()
    grid = GridSearchCV(pac, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best Parameters for Passive Aggressive Classifier:", grid.best_params_)
    return grid.best_estimator_

def plot_roc_pac(model, X_test, y_test, label_encoder):
    y_scores = model.decision_function(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

    plt.figure(figsize=(10, 8))
    for i in range(y_scores.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Passive Aggressive Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
