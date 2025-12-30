
import kagglehub
import pandas as pd
import os
import shutil

def download_data():
    """Download the heart disease dataset from Kaggle."""
    try:
        # Download latest version
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("nalisha/heart-disease-prediction-dataset")
        print("Path to dataset files:", path)
        
        # Determine destination
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest_dir = os.path.join(base_dir, 'data')
        
        # Check files in downloaded path
        files = os.listdir(path)
        csv_file = None
        for f in files:
            if f.endswith('.csv'):
                csv_file = f
                break
        
        if csv_file:
            src_path = os.path.join(path, csv_file)
            dest_path = os.path.join(dest_dir, 'heart_disease.csv')
            shutil.copy2(src_path, dest_path)
            print(f"Dataset copied to {dest_path}")
            return dest_path
        else:
            print("No CSV file found in downloaded dataset")
            return None
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_data(filepath=None):
    """Load the dataset into a pandas DataFrame."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'heart_disease.csv')
    
    if not os.path.exists(filepath):
        print("Dataset not found locally, attempting download...")
        filepath = download_data()
        if not filepath:
            raise FileNotFoundError("Could not find or download dataset")
    
    return pd.read_csv(filepath)
