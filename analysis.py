# pands-project.py
# author: Joseph Benkanoun
# analysis of the well-known Fisher’s Iris data set

# Importing modules for use in analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

# Importing Iris dataset from https://archive.ics.uci.edu/dataset/53/iris
from ucimlrepo import fetch_ucirepo 
iris = fetch_ucirepo(id=53) 
  
# Data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
  
# Combine data into one dataframe
iris_df = pd.concat([x, y], axis=1)

# Checking data types
print(iris_df.dtypes)

# Checking Class counts
print(iris_df["class"].value_counts())

# Cleaning Class column to make it a bit more legible
iris_df["class"] = iris_df["class"].replace({'Iris-setosa': 'Setosa', 'Iris-versicolor': 'Versicolor', 'Iris-virginica': 'Virginica'})

# Checking Class counts
print(iris_df["class"].value_counts())

# Export dataframe as CSV for review
iris_df.to_csv('iris.csv', index=False)

# Create file, will use to write overview to
# https://www.w3schools.com/python/python_file_write.asp
textfile = open("Iris Dataset Overview.txt", "w")
textfile.write("Iris Dataset Overview\n*********************\n\nThis file contains a preliminary overview of some of the key data points in Fisher's Iris flowers dataset.\n")