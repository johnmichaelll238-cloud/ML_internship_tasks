from src.config import RANDOM_STATE, LINEAR_KERNEL
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


def train_model(
    X_train, 
    y_train
    ):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    linear_SVM = SVC(
    kernel=LINEAR_KERNEL,
    random_state=RANDOM_STATE
    )
    linear_SVM.fit(X_train_tfidf, y_train)

    return linear_SVM, vectorizer
    
