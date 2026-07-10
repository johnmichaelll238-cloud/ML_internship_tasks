from tensorflow import keras
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.ops.metrics_impl import accuracy

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    MODEL_PATH
)

def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def train_models(model, X_train, y_train):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    return history
    

def save_model(model):
    model.save(MODEL_PATH)
    