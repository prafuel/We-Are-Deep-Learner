import pickle
import numpy as np

model = pickle.load(open("number_classifier.pkl", "rb"))

def classifiy_number(sample):
    sample = sample.reshape(1, 784)
    result = model.predict(sample)
    return np.argmax(result)