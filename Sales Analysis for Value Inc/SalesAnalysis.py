# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

data = pd.read_csv('transaction.csv', sep = ';')

data.info()

CostPerItem = data['CostPerItem']
NumberOfItemsPurchased = data['NumberOfItemsPurchased']
CostPerTransaction = CostPerItem * NumberOfItemsPurchased 

data['CostPerTransaction'] = CostPerTransaction

data['SalesPerTransaction'] = data['SellingPricePerItem'] * data['NumberOfItemsPurchased']

data['ProfitPerTransaction'] = data['SalesPerTransaction'] - data['CostPerTransaction']

data['Markup'] = data['ProfitPerTransaction']/data['CostPerTransaction']

data['Markup'] = round(data['Markup'], 2)

data['Date'] = data['Day'].astype(str) + '-' + data['Month'].astype(str) + '-' +data['Year'].astype(str)

split_col = data['ClientKeywords'].str.split(',',expand=True)

data['ClientAge'] = split_col[0]
data['ClientType'] = split_col[1]
data['LengthofContract'] = split_col[2]

data['ClientAge'] = data['ClientAge'].str.replace('[','') 

data['LengthofContract'] = data['LengthofContract'].str.replace(']','') 

data['ItemsDescription'] = data['ItemDescription'].str.lower()

seasons = pd.read_csv('value_inc_seasons.csv', sep = ';')

data=pd.merge(data ,seasons, on = 'Month')

data=data.drop(['Day','Month','Year','ClientKeywords'], axis = 1)

data.to_csv('ValueInc_Cleaned.csv', index = False)
