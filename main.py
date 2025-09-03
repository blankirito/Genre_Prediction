from data_loader import load_data, explore_data, clean_data
from preprocessing import (
    feature_engineering, apply_clean_text,
    filter_and_encode_genres, generate_tfidf_matrix
)
from balancing import balance_classes

# Load and clean data
df = load_data()
explore_data(df)
df = clean_data(df)
df = feature_engineering(df)
df = apply_clean_text(df)
df_filtered, le = filter_and_encode_genres(df)
tfidf_matrix, tfidf = generate_tfidf_matrix(df_filtered)

# Balance classes
X_resampled, y_resampled = balance_classes(tfidf_matrix, df_filtered['genre_label'])

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Logistic Regression
from logistic_model import train_logistic, evaluate_logistic, tune_logistic, plot_roc_logistic
best_log_reg = tune_logistic(X_train, y_train)
evaluate_logistic(best_log_reg, X_test, y_test, le)
plot_roc_logistic(best_log_reg, X_test, y_test, le)

# Random Forest
from random_forest_model import train_rf, evaluate_rf, tune_rf, plot_roc_rf
best_rf = tune_rf(X_train, y_train)
evaluate_rf(best_rf, X_test, y_test, le)
plot_roc_rf(best_rf, X_test, y_test, le)

# Multinomial Naive Bayes
from nb_model import train_nb, evaluate_nb, tune_nb, plot_roc_nb
best_nb = tune_nb(X_train, y_train)
evaluate_nb(best_nb, X_test, y_test, le)
plot_roc_nb(best_nb, X_test, y_test, le)

# Support Vector Classifier
from svc_model import train_svc, evaluate_svc, tune_svc, plot_roc_svc
best_svc = tune_svc(X_train, y_train)
evaluate_svc(best_svc, X_test, y_test, le)
plot_roc_svc(X_train, y_train, X_test, y_test, le)

# Passive Aggressive Classifier
from pac_model import train_pac, evaluate_pac, tune_pac, plot_roc_pac
best_pac = tune_pac(X_train, y_train)
evaluate_pac(best_pac, X_test, y_test, le)
plot_roc_pac(best_pac, X_test, y_test, le)

# Visualization
from visualization import (
    plot_model_accuracies,
    plot_logistic_feature_importance,
    plot_rf_feature_importance,
    plot_nb_feature_importance,
    plot_svc_feature_importance,
    plot_pac_feature_importance
)
from sklearn.metrics import accuracy_score

plot_model_accuracies(best_log_reg, best_rf, best_nb, best_svc, best_pac, X_test, y_test, accuracy_score)
plot_logistic_feature_importance(tfidf, best_log_reg)
plot_rf_feature_importance(tfidf, best_rf)
plot_nb_feature_importance(tfidf, best_nb)
plot_svc_feature_importance(tfidf, best_svc)
plot_pac_feature_importance(tfidf, best_pac)

# Recommendation System
from recommendation import (
    build_cosine_similarity_matrix,
    get_recommendation,
    get_recommendation_filtered,
    predict_genre_from_text
)

cosine_sim = build_cosine_similarity_matrix(tfidf_matrix)
print(get_recommendation("Breaking Bad", df_filtered, cosine_sim))
print(get_recommendation_filtered("Narcos", df_filtered, cosine_sim, top_n=5, genre_filter="Crime"))
print(predict_genre_from_text("A story about friendship and betrayal in a criminal world.", best_log_reg, tfidf, le))

# Clustering
from clustering import perform_kmeans_clustering, print_cluster_samples, plot_cluster_distribution
cluster_labels = perform_kmeans_clustering(tfidf_matrix, n_clusters=10)
df_filtered['cluster'] = cluster_labels

print_cluster_samples(df_filtered, cluster_labels)
plot_cluster_distribution(df_filtered)