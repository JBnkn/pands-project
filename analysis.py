# pands-project.py
# author: Joseph Benkanoun
# analysis of the well-known Fisher’s Iris data set

# Importing modules for use in analysis
import pandas as pd # for data manipulation and analysis
import numpy as np # 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
import warnings # to ignore Seaborn pairplot warning 'FutureWarning: use_inf_as_na'
warnings.filterwarnings('ignore')

# Set Seaborn visual theme for plots
sns.set()

# Importing Iris dataset from https://archive.ics.uci.edu/dataset/53/iris
from ucimlrepo import fetch_ucirepo 
iris = fetch_ucirepo(id=53) 
  
# Data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
  
# Combine data into one dataframe
iris_df = pd.concat([x, y], axis=1)

# Export dataframe as CSV for review
iris_df.to_csv('iris.csv', index=False)

# Create textfile https://www.w3schools.com/python/python_file_write.asp
# Will run some code to write overview data into textfile
textfile = open("Iris Dataset Overview.txt", "w")
textfile.write("Iris Dataset Overview\n*********************\n\n"
               "This file contains a preliminary overview of some of the key data points in Fisher's Iris flowers dataset.\n"
               "I will begin by taking a look at the data structure and data types.\n\n"
               )
textfile.write("The head() function will show me the first five rows of the dataframe:\n")
textfile.write(str(iris_df.head()))
textfile.write("\n\ndtypes will show me the data types within the dataframe:\n")
textfile.write(str(iris_df.dtypes))
textfile.write("\n\nI can see that we have four measurement variables that are Float64 values, and a Class variable that refers to the type of flower.\n"
               "I will now use the describe method to get an overview of the data within the quantitative variables.\n\n")
textfile.write(str(iris_df.describe()))
textfile.write("\n\nBased on the counts, there are 150 flowers overall in this dataset. I'll use value_counts to check the distribution across the Class variable.\n\n")
textfile.write(str(iris_df["class"].value_counts()))
textfile.write("\n\nLooks like the flowers are evenly distributed across three classes. I find the formatting of the class names a bit distracting so I'm going to tidy them slightly.")

# Cleaning Class column to make it a bit more legible
iris_df["class"] = iris_df["class"].replace({'Iris-setosa': 'Setosa', 'Iris-versicolor': 'Versicolor', 'Iris-virginica': 'Virginica'})

# Confirming Class column has been updated
textfile.write("\n\nLet's double-check to make sure that it worked:\n\n")
textfile.write(str(iris_df["class"].value_counts()))
textfile.write("\n\nPerfect! Just what I wanted.\n\n")

# Creating histograms for all four quantitative variables to see distribution
# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
axs[0,0].hist(iris_df['sepal length']) # Plotting the histogram for Sepal Length
axs[0,0].set_title('Sepal Length (mm)', fontsize=12, fontweight='bold')
axs[0,1].hist(iris_df['sepal width']) # Plotting the histogram for Sepal Width
axs[0,1].set_title('Sepal Width (mm)', fontsize=12, fontweight='bold')
axs[1,0].hist(iris_df['petal length']) # Plotting the histogram for Petal Length
axs[1,0].set_title('Petal Length (mm)', fontsize=12, fontweight='bold')
axs[1,1].hist(iris_df['petal width']) # Plotting the histogram for Petal Width
axs[1,1].set_title('Petal Width (mm)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Iris - distribution of quantitative variables")

# Creating scatter plot for Sepal Length v Width, using Class as hue
plt.clf()
plt.figure(figsize=(10,10))
sns.scatterplot(x='sepal length', y='sepal width', hue='class', data=iris_df)
plt.title("Sepal Length v Sepal Width")
plt.savefig("Iris - scatterplot Sepal Length v Sepal Width")

# Creating scatter plot for Petal Length v Width, using Class as hue
plt.clf()
sns.scatterplot(x='petal length', y='petal width', hue='class', data=iris_df)
plt.title("Petal Length v Petal Width")
plt.savefig("Iris - scatterplot Petal Length v Petal Width")

# Creating Seaborn pairplot to look at scatter plots across all variables with Class as hue
sns.pairplot(iris_df, hue="class")
plt.savefig("Iris - scatterplot Seaborn pairplot")

# Create separate dataframes for each class
setosa_df = iris_df[iris_df['class'] == 'Setosa']
versicolor_df = iris_df[iris_df['class'] == 'Versicolor']
virginica_df = iris_df[iris_df['class'] == 'Virginica']

# Sending some further overview to textfile
# Will look at key data points for each class
textfile.write("Let's look at the key data points for each class of flower. Mean will tell us the average value, while Median will show us the middle value.\n\n")
textfile.write("Setosa:\n"
               f"Sepal Length - Mean: {round(setosa_df['sepal length'].mean(),2)} Median: {setosa_df['sepal length'].median()}\n"
               f"Sepal Width  - Mean: {round(setosa_df['sepal width'].mean(),2)} Median: {setosa_df['sepal width'].median()}\n"
               f"Petal Length - Mean: {round(setosa_df['petal length'].mean(),2)} Median: {setosa_df['petal length'].median()}\n"
               f"Petal Width  - Mean: {round(setosa_df['petal width'].mean(),2)} Median: {setosa_df['petal width'].median()}\n\n"
               "Versicolor:\n"
               f"Sepal Length - Mean: {round(versicolor_df['sepal length'].mean(),2)} Median: {versicolor_df['sepal length'].median()}\n"
               f"Sepal Width  - Mean: {round(versicolor_df['sepal width'].mean(),2)} Median: {versicolor_df['sepal width'].median()}\n"
               f"Petal Length - Mean: {round(versicolor_df['petal length'].mean(),2)} Median: {versicolor_df['petal length'].median()}\n"
               f"Petal Width  - Mean: {round(versicolor_df['petal width'].mean(),2)} Median: {versicolor_df['petal width'].median()}\n\n"
               "Virginica:\n"
               f"Sepal Length - Mean: {round(virginica_df['sepal length'].mean(),2)} Median: {virginica_df['sepal length'].median()}\n"
               f"Sepal Width  - Mean: {round(virginica_df['sepal width'].mean(),2)} Median: {virginica_df['sepal width'].median()}\n"
               f"Petal Length - Mean: {round(virginica_df['petal length'].mean(),2)} Median: {virginica_df['petal length'].median()}\n"
               f"Petal Width  - Mean: {round(virginica_df['petal width'].mean(),2)} Median: {virginica_df['petal width'].median()}\n\n"
               )

