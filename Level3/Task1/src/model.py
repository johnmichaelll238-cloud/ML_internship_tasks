from src.config import MODEL_PATH, VECTOR_PATH
import joblib


def save_model(svm, vectorizer):
    joblib.dump(svm, MODEL_PATH)
    joblib.dump(vectorizer, VECTOR_PATH)
    
    print("MODELS SERIALISED SUCCESSFULLY")