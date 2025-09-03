from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def balance_classes(X, y, random_state=42, verbose=True):
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    if verbose:
        print("Original dataset shape:", pd.Series(y).value_counts())
        print("Resampled dataset shape:", pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled
