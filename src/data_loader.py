# src/data_loader.py
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse


class CustomDataset(Dataset): 
    def __init__(self, train:bool): 
        self.train = train
        if self.train: 
            self.data, self.target = self.get_train_data()
        # transformations are stored in the functions
            self.data_transformed = self.get_data_transform()
            self.target_transformed = self.get_target_transform()
        else:
            self.data = self.get_test_data()
            self.data_transformed = self.get_data_transform()
        
    def get_train_data(self): 
        self.df_kidd = pd.read_csv('data/kidd_train.csv')
        self.df_seldin = pd.read_csv('data/seldin_train.csv')

        self.df = pd.merge(
            self.df_kidd, 
            self.df_seldin,
            on=['id', 'gender', 'superpopulation'],
            suffixes=('_kidd', '_seldin')
        )
        data = self.df.dropna(subset=['superpopulation'])
        data = self.df.drop(columns=['id','superpopulation'])
        target = self.df['superpopulation'].values

        return data, target
    
    def get_test_data(self):
        self.df_kidd = pd.read_csv('data/kidd_test.csv')
        self.df_seldin = pd.read_csv('data/seldin_test.csv')
        self.df = pd.merge(
            self.df_kidd, 
            self.df_seldin,
            on=['id', 'gender'],
            suffixes=('_kidd', '_seldin')
        )
        data = self.df.drop(columns=['id'])
        return data
    
    def get_data_transform(self) -> torch.tensor:
        imputer = SimpleImputer(strategy='most_frequent')
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.pipeline = Pipeline(
            steps=[
                ('imputer', imputer), 
                ('encoder', encoder)
            ]
        )
        X = self.pipeline.fit_transform(self.data)
        X = X.toarray() if issparse(X) else X
        return torch.tensor(X, dtype=torch.float32)

    def get_target_transform(self) -> torch.tensor:
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.target)
        return torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.train: 
            return self.data_transformed[index], self.target_transformed[index]
        else:
            return self.data_transformed[index]



