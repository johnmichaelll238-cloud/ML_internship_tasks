import pandas as pd
from src.config import DATASET_PATH, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
    
    df = df[["Text", "Sentiment"]]
    
    class_counts = df["Sentiment"].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    df = df[~df["Sentiment"].isin(rare_classes)]

    X = df["Text"]
    y = df["Sentiment"]
    
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
    )
    return(
        X_train,
        X_test, 
        y_train, 
        y_test
    )
def get_counts():
    df = pd.read_csv(DATASET_PATH)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.lower()
    
    df = df[["Text", "Sentiment"]]    
    
    class_counts = df["Sentiment"].value_counts()
    return class_counts
    