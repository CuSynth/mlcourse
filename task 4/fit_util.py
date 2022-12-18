import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dataTransformer import DataTransformer


def load_data():
    train_df: pd.DataFrame = pd.read_csv("data/train.csv")

    transformer = DataTransformer()
    train_df = transformer.fit_transform(train_df)

    target = np.log1p(train_df['SalePrice'])
    train_df: pd.DataFrame = train_df.drop(columns='SalePrice', axis=1)

    return train_df, target



def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def evaluate(model, X, y, msg: str = ""):
    preds = model.predict(X)
    print(msg + "RMSE: " + str(rmse(preds, y)))
    


def to_categorical(X):
    for c in X.columns:
        col_type = X[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('category')


def submission(transformer, gs_model, create_sub_file: bool=False, obj_to_num: bool=True):
    submissions = pd.read_csv("data/sample_submission.csv")
    validation = pd.read_csv("data/test.csv")

    val_ids = validation["Id"]
    validation = validation.drop(columns=["Id"])

    validation['GrLivArea'] = np.log1p(validation['GrLivArea'])
    validation['TotalBsmtSF'] = np.log1p(validation['TotalBsmtSF'])
    validation = pd.get_dummies(validation, drop_first=True)

    sub_predictions = gs_model.predict(validation)
    print("RMSE submission: " + str(rmse(sub_predictions, np.log1p(submissions["SalePrice"]))))
    
    if create_sub_file:
        d = {'Id': val_ids.to_numpy(), 'SalePrice':  np.expm1(sub_predictions)}
        df = pd.DataFrame(data=d)
        df.to_csv('submission.csv', index=False)