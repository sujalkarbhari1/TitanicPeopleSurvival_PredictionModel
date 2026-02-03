from config import Config
# Import Data Manipulation Libraries
import pandas as pd
import numpy as np


# Data Ingestion

def data_ingestion():
    df = pd.read_csv(Config.filepath)
    return df


