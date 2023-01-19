# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:46:15 2023

@author: franc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import describe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def readfile(doc, columns, indicator):
    filedata = pd.read_excel(doc, skiprows=3)
    filedata = filedata.loc[filedata['Indicator Name'] == indicator]
    filedata = filedata[columns]
    filedata.set_index('Country Name', inplace  = True)
    return filedata, filedata.T

doc =r"C:\Users\franc\OneDrive\Desktop\WBDATA.xls"
Indicator ='Electricity production from oil sources (% of total)'
columns = ['Country Name', '1985', '2015']

WB_DATA = readfile(doc, columns, Indicator)
print(WB_DATA)
x =WB_DATA[0]
x_c= x.dropna()
x_a =x_c.values
print(x_a)

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x_a)
print(scaled_x)
