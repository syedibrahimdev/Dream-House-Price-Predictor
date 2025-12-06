import pandas as pd
import numpy as np

def engineer_features(df):
    data = df.copy()
    
    cols_to_fill_0 = ['MasVnrArea', 'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    for col in cols_to_fill_0:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    full_bath = data.get('FullBath', 0)
    half_bath = data.get('HalfBath', 0)
    bsmt_full = data.get('BsmtFullBath', 0)
    bsmt_half = data.get('BsmtHalfBath', 0)
    data['TotalBath'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']

    data['TotalPorchSF'] = data[porch_cols].sum(axis=1) if set(porch_cols).issubset(data.columns) else 0

    if 'YearBuilt' in data.columns:
        data['YearBuilt'] = data['YearBuilt'].fillna(2000).astype(int)
    if 'YearRemodAdd' in data.columns:
        data['YearRemodAdd'] = data['YearRemodAdd'].fillna(2000).astype(int)
    if 'YrSold' in data.columns:
        data['YrSold'] = data['YrSold'].fillna(2010).astype(int)
        
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
    
    drop_cols = ['Id', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    
    return data