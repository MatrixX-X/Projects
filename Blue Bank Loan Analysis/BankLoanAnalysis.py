# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:16:38 2022

@author: Abdul Mateen
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#method 1 to read json file
json_file = open('loan_data_json.json')
data = json.load(json_file)

#method 2 to read json file
with open('loan_data_json.json') as json_file:
    data = json.load(json_file)
    
#transform to dataframe

loandata = pd.DataFrame(data)

loandata['purpose'].unique()

loandata.describe()

loandata['int.rate'].describe()
loandata['fico'].describe()
loandata['dti'].describe()

# using EXP() to get the annual income
income = np.exp(loandata['log.annual.inc'])
loandata['annualincome'] = income

ficocat = []
for x in range(0,len(loandata)):
    category = loandata['fico'][x]
    
    try:
        
        if category>=300 and category<400:
            cat = 'Very poor'
        elif category>=400 and category<600:
            cat = 'Poor'
        elif category>=600 and category<660:
            cat = 'Fair'
        elif category>=660 and category<700:
            cat = 'Good'
        elif category>=700:
            cat = 'Excellent'
        else:
            cat = 'Unknown'
    except:
        cat = 'Unknown'
    ficocat.append(cat)

ficocat = pd.Series(ficocat)

loandata['fico.category'] = ficocat

# another way for conditional statements

loandata.loc[loandata['int.rate'] > 0.12, 'int.rate.type'] = 'High'
loandata.loc[loandata['int.rate'] <= 0.12, 'int.rate.type'] = 'Low'

catplot = loandata.groupby(['fico.category']).size()
catplot.plot.bar()
plt.show()
# size number of rows

purposecount = loandata.groupby(['purpose']).size()
purposecount.plot.bar()
plt.show()

ypoint = loandata['annualincome']
xpoint = loandata['dti']
plt.scatter(xpoint,ypoint)
plt.show()

loandata.to_csv('loan_cleaned.csv', index = True)

