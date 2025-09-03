import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_accuracies(best_log_reg, best_rf, best_nb, best_svc, best_pac, X_test, y_test, accuracy_score, model_names=None):
    acc_logreg = accuracy_score(y_test, best_log_reg.predict(X_test))
    acc_rf = accuracy_score(y_test, best_rf.predict(X_test))
    acc_nb = accuracy_score(y_test, best_nb.predict(X_test))
    acc_svc = accuracy_score(y_test, best_svc.predict(X_test))
    acc_pac = accuracy_score(y_test, best_pac.predict(X_test))

    models = model_names if model_names else ['Logistic Regression', 'Random Forest', 'Multinomial NB', 'Support Vector Classifier', 'Passive Aggressive Classifier']
    accuracies = [acc_logreg, acc_rf, acc_nb, acc_svc, acc_pac]

    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_logistic_feature_importance(tfidf_vectorizer, logistic_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coefficients = logistic_model.coef_
    avg_coefficients = np.mean(np.abs(coefficients), axis=0)

    feat_imp_lr = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_coefficients
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_lr['feature'].head(20)[::-1], feat_imp_lr['importance'].head(20)[::-1], color='blue')
    plt.xlabel('Coefficient Magnitude')
    plt.title('Top 20 Important Features from Logistic Regression')
    plt.tight_layout()
    plt.show()


def plot_rf_feature_importance(tfidf_vectorizer, rf_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_

    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_df['feature'].head(20)[::-1], feat_imp_df['importance'].head(20)[::-1], color='green')
    plt.xlabel('Importance Score')
    plt.title('Top 20 Important Features from Random Forest')
    plt.tight_layout()
    plt.show()


def plot_nb_feature_importance(tfidf_vectorizer, nb_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    class_log_probs = nb_model.feature_log_prob_
    avg_log_probs = np.mean(class_log_probs, axis=0)

    feat_imp_nb = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_log_probs
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_nb['feature'].head(20)[::-1], feat_imp_nb['importance'].head(20)[::-1], color='orange')
    plt.xlabel('Average Log Probability')
    plt.title('Top 20 Important Features from Multinomial Naive Bayes')
    plt.tight_layout()
    plt.show()

def plot_svc_feature_importance(tfidf_vectorizer, svc_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()

    coefficients = svc_model.coef_ 

    avg_coefficients = np.mean(np.abs(coefficients), axis=0)

    feat_imp_svc = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_coefficients
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_svc['feature'].head(20)[::-1], feat_imp_svc['importance'].head(20)[::-1], color='red')
    plt.xlabel('Coefficient Magnitude')
    plt.title('Top 20 Important Features from Support Vector Classifier')
    plt.tight_layout()
    plt.show()
    
def plot_pac_feature_importance(tfidf_vectorizer, pac_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()

    coef = pac_model.coef_

    importance = np.mean(np.abs(coef), axis=0)

    feat_imp_pac = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_pac['feature'].head(20)[::-1], feat_imp_pac['importance'].head(20)[::-1], color='teal')
    plt.xlabel('Importance Score (Average Absolute Coefficient)')
    plt.title('Top 20 Important Features from PassiveAggressiveClassifier')
    plt.tight_layout()
    plt.show()