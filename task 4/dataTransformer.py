import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataTransformer:
    
    def __init__(self):
        self.encoder = LabelEncoder()


        
    def fillnans(self, X: pd.DataFrame, train_set: bool=True):
        # total = X.isnull().sum().sort_values(ascending=False)
        # X = X.drop((total[total > 1]).index, axis=1)

        X =X.drop(X.loc[X['Electrical'].isnull()].index)

        todrop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                  'LotFrontage', 
                  'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual',
                  'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 
                  'MasVnrArea', 'MasVnrType']

        X = X.drop(columns=todrop, axis=1)



        to_fill = [ 'MSZoning', 'BsmtHalfBath', 'Utilities',
                   'Functional', 'BsmtFullBath', 'BsmtFinSF1', 
                   'BsmtFinSF2', 'BsmtUnfSF', 'KitchenQual', 'TotalBsmtSF',
                   'Exterior2nd', 'GarageCars', 'Exterior1st',
                   'SaleType']
        for elem in to_fill:
            X[elem] = X[elem].fillna(X[elem].mode()[0])

        zero_nan_cols = ['GarageArea']
        X[zero_nan_cols] = X[zero_nan_cols].fillna(0)

        return X
                
    def encode(self, X):
        object_candidates = list(X.dtypes[X.dtypes == "object"].index.values)
        
        for col in object_candidates:
            X[col] = self.encoder.fit_transform(X[col])
        
        return X

    # def encode(self, X):
    #     X = pd.get_dummies(X, drop_first=True)
    #     return X
        
    def fit_transform(self, X, obj_to_num: bool=True, train_set: bool=True, drop_id: bool=True):
        X = self.fillnans(X)

        if train_set:
            X.sort_values(by = 'GrLivArea', ascending = False)[:2]
            X = X.drop(X[X['Id'] == 1299].index)
            X = X.drop(X[X['Id'] == 524].index)

            X['SalePrice']= np.log1p(X['SalePrice'])
        
        X['GrLivArea'] = np.log1p(X['GrLivArea'])
        X['TotalBsmtSF'] = np.log1p(X['TotalBsmtSF'])

        if drop_id:
            X = X.drop("Id", axis=1)
            
        if obj_to_num:
            X = self.encode(X)
        
        return X     