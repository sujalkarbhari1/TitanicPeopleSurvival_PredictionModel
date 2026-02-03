from config import Config
# Import Data Manipulation libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler,LabelEncoder
from imblearn.over_sampling import SMOTE

def data_preprocessing(df):
    
    df = pd.read_csv(Config.filepath)

    # Drop Unnecessary Columns
    df.drop(columns = ['PassengerId','CabinDeck','Ticket','Title','AgeGroup'],inplace=True,axis =1)
    # Handling Leakage
    '''
    1. Split the data into X and y
    2. Use Train Test Split
    3. Using LabelEncoder
    4. Using Scaling Technique
    5. Using SMOT Technique for data balance
    '''
    # Split the Data into X and y
    X = df.drop(columns='Survived',axis=1)
    y = df['Survived']

    # Use Train Test Split
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=1)

    # Segregate the Numerical and Categorical Columns
    numerical_col = X_train.select_dtypes(exclude='object').columns
    categorical_col = X_train.select_dtypes(include='object').columns

    # Apply Label Encoding on Categorical Columns
    le = LabelEncoder()
    for i in categorical_col:
        X_train[i] = le.fit_transform(X_train[i])  # Seen Data
        X_test[i] = le.transform(X_test[i])           # Unseen Data

    # Using Scaling techiques on Numerical Columns
    sc = RobustScaler()
    X_train[numerical_col] = sc.fit_transform(X_train[numerical_col]) # Seen data
    X_test[numerical_col] = sc.transform(X_test[numerical_col]) # Unseen Data

    # Using SMOTE
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train) # Seen data

    return X_train,X_test,y_train,y_test