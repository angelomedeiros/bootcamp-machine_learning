#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv("credit-data.csv")
stat = base.describe()
negatives_age = base.loc[base['age'] < 0]

# Remove coluna inteira
# base.drop('age', 1, inplace=True)

# Remove registros
# base.drop(base[base.age < 0].index, inplace = True)

# Substituir valores
base.mean()
base['age'].mean()
mean_positive_ages = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = mean_positive_ages

pd.isnull(base['age'])
# base.loc[pd.isnull(base['age'])] = mean_positive_ages

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)