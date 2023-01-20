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
import scipy.optimize as opt

from sklearn.metrics import silhouette_score
import err_ranges as err
from scipy.stats import describe
from typing import List, Tuple

#Function to read excel file and return both original and transposed file
def readfile(doc:str, columns:List[str], indicator:str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Reads an excel file and returns the specified indicator data with specified columns and country name as index.

    Parameters:
    - doc (str): path to excel file
    - columns (List[str]): list of columns to select
    - indicator (str): indicator name to filter data
    
    Returns:
    - Tuple of two dataframes. The first dataframe contains the filtered data with 'Country Name' as the index column,
      the second dataframe is the transpose of the first dataframe
    """
    WB_data = pd.read_excel(doc, skiprows=3)
    WB_data = WB_data.loc[WB_data['Indicator Name'] == indicator]
    WB_data = WB_data[columns]
    WB_data.set_index('Country Name', inplace  = True)
    return WB_data, WB_data.T


doc =r"C:\Users\franc\OneDrive\Desktop\WBDATA.xls"
Indicator ='Electricity production from oil sources (% of total)'
columns = ['Country Name', '1985', '2015']

WB_DATA = readfile(doc, columns, Indicator)

# Assign first element of tuple (filtered data) to data
data = WB_DATA[0] 

 # Drop any rows with missing values
data1 = data.dropna()

# convert the dataframe to numpy array
data1_arr = data1.values 

# Initialize a StandardScaler object
scaler = StandardScaler() 

 #Scale the data
scaled_x = scaler.fit_transform(data1_arr) 
print(scaled_x)




# Initialize a list to store the sum of squared errors (SSE) for each cluster
sse = []

# Loop through the number of clusters
for i in range(1, 11):
    # Initialize the KMeans model with specified number of clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Fit the model to the data
    kmeans.fit(data1_arr)
    # Append the SSE to the list
    sse.append(kmeans.inertia_)

# Plot the SSE for each cluster
plt.plot(range(1, 11), sse)
# Save the plot as an image file
plt.savefig('clusters.png')
# Show the plot
plt.title('K-MEAN ELBOW')
plt.xlabel('No of Clusters')
plt.ylabel('SSE')
plt.show()

# Initialize the KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# Fit the model to the data and predict the cluster labels for each data point
y_kmeans = kmeans.fit_predict(data1_arr)
# Print the cluster labels
print(y_kmeans)

# Add a new column 'Cluster' to the dataframe with the predicted cluster labels
data1['Cluster'] = y_kmeans

# Assign the cluster centers to a variable
Centroids = kmeans.cluster_centers_

# Plot the data points
plt.scatter(data.iloc[:,0], data.iloc[:,1], s=50, c='black')
# Plot the data points
plt.scatter(data.iloc[:,0], data.iloc[:,1], s=50, c='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, c = 'red', label = 'Centroids')
# Add a legend to the plot
plt.legend()
# Show the plot
plt.show()


z = data.values


# Plot the data points for each cluster
plt.scatter(data1_arr[y_kmeans == 0, 0],data1_arr[y_kmeans == 0, 1], s = 50, c = 'purple',label = 'Cluster 1')
plt.scatter(data1_arr[y_kmeans == 1, 0], data1_arr[y_kmeans == 1, 1], s = 50, c = 'orange',label = 'Cluster 2')
plt.scatter(data1_arr[y_kmeans == 2, 0], data1_arr[y_kmeans == 2, 1], s = 50, c = 'green',label = 'Cluster 3')

# Add title to the plot
plt.title('Clusters of Countries based on Electricity Production from Oil Sources')

# Plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 50, c = 'red', label = 'Centroids')
# Add a legend to the plot


plt.legend()

# Show the plot
plt.show()

#show silhouette score to show similarity of the objects within the cluster
score = silhouette_score(data1_arr, y_kmeans)
score


#Create new dataframes for each clusters
df_cluster1 = data1.loc[data1['Cluster'] == 0]
df_cluster2 = data1.loc[data1['Cluster'] == 1]
df_cluster3 = data1.loc[data1['Cluster'] == 2]


#Data Curve Fitting----------------------------------------------

def read_data(file_path):
    """
    Reads data from an excel file and returns a DataFrame
    
    Parameters:
    file_path (str): the file path of the excel file
    
    Returns:
    DataFrame: the data from the excel file
    """
    df2 = pd.read_excel(file_path, skiprows=3)
    return df2
    
data2 = read_data(r"C:\Users\franc\OneDrive\Desktop\WBDATA.xls")


#drop columns
data2 =data2.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)

#transpose of original data
data_year = data2.T

#rename the columns
data_year = data_year.rename(columns=data_year.iloc[0])

#drop the country name
data_year = data_year.drop(index=data_year.index[0], axis=0)
data_year['Year'] = data_year.index


#convert the 'Year' and 'Germany' columns to numeric values
data_fitting = data_year[['Year', 'Germany']].apply(pd.to_numeric, 
                                               errors='coerce')

#drop any missing values from the dataframe
curve = data_fitting.dropna(axis=1).values

#Assign the first column to x_axis and the second column to y_axis
x_axis = curve[:,0]
y_axis = curve[:,1]
print(y_axis)

# The curve_fit function is used to fit the model to the data. The first argument is the model function, and the second and third 
# arguments are the independent and dependent variables respectively. The popt variable stores the optimal parameter values.
def model(x, a, b, c, d):
    '''
    This function defines the model that will be used for curve fitting. It takes in the independent variable x, and the parameters 
    a, b, c and d. The function returns the value of the model for a given x and parameter values.
    '''
    return a*x**3 + b*x**2 + c*x + d


# The curve_fit function also returns the covariance matrix of the parameters
popt, _ = opt.curve_fit(model, x_axis, y_axis)


param, covar = opt.curve_fit(model, x_axis, y_axis)

# Unpacking the optimal parameter values
a, b, c, d = popt

# This line of code creates a plot of the log absolute values of the covariance matrix
plt.imshow(np.log(np.abs(covar)))
plt.colorbar()
plt.show()

# This line of code creates a scatter plot of the data
plt.scatter(x_axis, y_axis)

# This line of code creates an array of x values and calculates the corresponding y values using the optimal parameter values
x_line = np.arange(min(curve[:,0]), max(curve[:,0])+1, 1)
y_line = model(x_line, a, b, c, d)

# This line of code creates a scatter plot of the data and a line plot of the fitted model
plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line, '--', color='black', label='Fitted Model')
plt.xlabel('Year')
plt.ylabel('Germany')
plt.title('CURVE FITTING')
plt.legend()
plt.show()

# This line of code calculates the standard deviation of the parameters
sigma = np.sqrt(np.diag(covar))

# This line of code calculates the lower and upper bounds of the error for each x value
low, up = err.err_ranges(x_axis, model, popt, sigma)
print(low, up)

# This line of code creates a scatter plot of the data, a line plot of the fitted model and fills the area between the lower and upper bounds of the error
print(curve)
print(low.shape)
plt.scatter(x_axis, y_axis)
plt.plot(x_line, y_line, '--', color='black')
plt.fill_between(x_axis, low, up, alpha=0.7, color='green')
plt.show()



