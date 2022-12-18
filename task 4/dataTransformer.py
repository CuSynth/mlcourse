import numpy as np
import pandas as pd


class DataTransformer:
    
    def __init__(self):
        pass

        
    def fillnans(self, X: pd.DataFrame):
        total = X.isnull().sum().sort_values(ascending=False)
        X = X.drop((total[total > 1]).index, axis=1)
        X = X.drop(X.loc[X['Electrical'].isnull()].index)

        return X
                
    def encode(self, X):
        X = pd.get_dummies(X, drop_first=True)
        return X
    
    def fit_transform(self, X, obj_to_num: bool=True, drop_ID: bool=True):
        X = self.fillnans(X)

        X.sort_values(by = 'GrLivArea', ascending = False)[:2]
        X = X.drop(X[X['Id'] == 1299].index)
        X = X.drop(X[X['Id'] == 524].index)

        X['SalePrice']= np.log1p(X['SalePrice'])
        X['GrLivArea'] = np.log1p(X['GrLivArea'])

        X['TotalBsmtSF'] = np.log1p(X['TotalBsmtSF'])

        if drop_ID:
            X = X.drop("Id", axis=1)

        if obj_to_num:
            X = self.encode(X)
            
        
        return X     