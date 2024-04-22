import logging
import pickle
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def load_pca_model(filepath):
    """Load the PCA model from a file."""
    try:
        with open(filepath, 'rb') as file:
            pca = pickle.load(file)
        if hasattr(pca, 'components_'):
            print("PCA model loaded and is fitted.")
            return pca
        else:
            print("Loaded PCA model is not fitted.")
            return None
    except FileNotFoundError:
        print("PCA model file not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading PCA model: {e}")
        return None