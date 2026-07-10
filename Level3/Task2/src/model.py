from src.config import HIDDEN_UNITS, INPUT_DIM, NUM_CLASSES
from tensorflow import keras

def build_model():
    model = keras.Sequential([
         keras.layers.Input(shape=(INPUT_DIM,)),
         
         keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
         
         keras.layers.Dense(NUM_CLASSES, activation="softmax")

    ])
    return model
    
