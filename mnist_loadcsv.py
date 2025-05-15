import numpy as np
import pandas as pd

def load_mnistcsv(path):
    df = pd.read_csv(path)
    data = df.iloc[:, 1:]
    data = Normalization(data)
    data_labels = df.iloc[:, 0]
    return np.array(data), np.array(data_labels)

# Normalize pixel values to [0, 1]
def Normalization(data):
    return data / 255.0