import pandas as pd

def load_data(filepath='netflix_titles.csv'):
    df = pd.read_csv(filepath)
    print("Initial Data Sample:")
    print(df.head())
    return df

def explore_data(df):
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nUnique Titles:", df['title'].nunique())
    print("\nType Distribution:\n", df['type'].value_counts())
    print("\nTop Genres:")
    print(df['listed_in'].value_counts().head(10))

def clean_data(df):
    df['description'] = df['description'].fillna('')
    df = df.dropna(subset=['title', 'listed_in'])
    df = df.reset_index(drop=True)
    print("Data Cleaned, Rows Remaining:", len(df))
    return df
