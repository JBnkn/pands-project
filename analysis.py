# pands-project.py
# author: Joseph Benkanoun
# analysis of the well-known Fisher’s Iris data set

# Importing modules for use in analysis
import pandas as pd # for dataframes and analysis
import numpy as np # for numerical operations
import seaborn as sns # for creating visualisations
import matplotlib.pyplot as plt # for plotting
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
textfile.write("The head() function will show me the first five rows of the dataframe:\n\n")
textfile.write(str(iris_df.head()))
textfile.write("\n\ndtypes will show me the data types within the dataframe:\n\n")
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
axs[0,0].hist(iris_df['petal length']) # Plotting the histogram for Petal Length
axs[0,0].set_title('Petal Length (mm)', fontsize=12, fontweight='bold')
axs[0,1].hist(iris_df['petal width']) # Plotting the histogram for Petal Width
axs[0,1].set_title('Petal Width (mm)', fontsize=12, fontweight='bold')
axs[1,0].hist(iris_df['sepal length']) # Plotting the histogram for Sepal Length
axs[1,0].set_title('Sepal Length (mm)', fontsize=12, fontweight='bold')
axs[1,1].hist(iris_df['sepal width']) # Plotting the histogram for Sepal Width
axs[1,1].set_title('Sepal Width (mm)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Iris - distribution of quantitative variables")

# Creating scatter plot for Petal Length v Width, using Class as hue
plt.clf()
plt.figure(figsize=(10,10))
sns.scatterplot(x='petal length', y='petal width', hue='class', data=iris_df)
plt.title("Petal Length v Petal Width")
plt.savefig("Iris - scatterplot Petal Length v Petal Width")

# Creating scatter plot for Sepal Length v Width, using Class as hue
plt.clf()
sns.scatterplot(x='sepal length', y='sepal width', hue='class', data=iris_df)
plt.title("Sepal Length v Sepal Width")
plt.savefig("Iris - scatterplot Sepal Length v Sepal Width")

# Creating Seaborn pairplot to look at scatter plots across all variables with Class as hue
sns.pairplot(iris_df, hue="class")
plt.savefig("Iris - scatterplot Seaborn pairplot")

# Create separate dataframes for each class
setosa_df = iris_df[iris_df['class'] == 'Setosa']
versicolor_df = iris_df[iris_df['class'] == 'Versicolor']
virginica_df = iris_df[iris_df['class'] == 'Virginica']

# Sending some further overview to textfile
# Will look at key data points for each class
textfile.write("Let's look at the key data points for each class of flower. Min will show the minimum value, Max the maximum, Mean the average value, and Median will show us the middle value.\n\n")
textfile.write("\tSetosa:\t\t\tMin\t\tMax\t\tMean\tMedian\n"
               f"\tPetal Length -\t{(setosa_df['petal length'].min()):.1f}\t\t{(setosa_df['petal length'].max()):.1f}\t\t{(setosa_df['petal length'].mean()):.1f}\t\t{(setosa_df['petal length'].median()):.1f}\n"
               f"\tPetal Width  -\t{(setosa_df['petal width'].min()):.1f}\t\t{(setosa_df['petal width'].max()):.1f}\t\t{(setosa_df['petal width'].mean()):.1f}\t\t{(setosa_df['petal width'].median()):.1f}\n"
               f"\tSepal Length -\t{(setosa_df['sepal length'].min()):.1f}\t\t{(setosa_df['sepal length'].max()):.1f}\t\t{(setosa_df['sepal length'].mean()):.1f}\t\t{(setosa_df['sepal length'].median()):.1f}\n"
               f"\tSepal Width  -\t{(setosa_df['sepal width'].min()):.1f}\t\t{(setosa_df['sepal width'].max()):.1f}\t\t{(setosa_df['sepal width'].mean()):.1f}\t\t{(setosa_df['sepal width'].median()):.1f}\n\n"
               "\tVersicolor:\n"
               f"\tPetal Length -\t{(versicolor_df['petal length'].min()):.1f}\t\t{(versicolor_df['petal length'].max()):.1f}\t\t{(versicolor_df['petal length'].mean()):.1f}\t\t{(versicolor_df['petal length'].median()):.1f}\n"
               f"\tPetal Width  -\t{(versicolor_df['petal width'].min()):.1f}\t\t{(versicolor_df['petal width'].max()):.1f}\t\t{(versicolor_df['petal width'].mean()):.1f}\t\t{(versicolor_df['petal width'].median()):.1f}\n"
               f"\tSepal Length -\t{(versicolor_df['sepal length'].min()):.1f}\t\t{(versicolor_df['sepal length'].max()):.1f}\t\t{(versicolor_df['sepal length'].mean()):.1f}\t\t{(versicolor_df['sepal length'].median()):.1f}\n"
               f"\tSepal Width  -\t{(versicolor_df['sepal width'].min()):.1f}\t\t{(versicolor_df['sepal width'].max()):.1f}\t\t{(versicolor_df['sepal width'].mean()):.1f}\t\t{(versicolor_df['sepal width'].median()):.1f}\n\n"
               "\tVirginica:\n"
               f"\tPetal Length -\t{(virginica_df['petal length'].min()):.1f}\t\t{(virginica_df['petal length'].max()):.1f}\t\t{(virginica_df['petal length'].mean()):.1f}\t\t{(virginica_df['petal length'].median()):.1f}\n"
               f"\tPetal Width  -\t{(virginica_df['petal width'].min()):.1f}\t\t{(virginica_df['petal width'].max()):.1f}\t\t{(virginica_df['petal width'].mean()):.1f}\t\t{(virginica_df['petal width'].median()):.1f}\n"
               f"\tSepal Length -\t{(virginica_df['sepal length'].min()):.1f}\t\t{(virginica_df['sepal length'].max()):.1f}\t\t{(virginica_df['sepal length'].mean()):.1f}\t\t{(virginica_df['sepal length'].median()):.1f}\n"
               f"\tSepal Width  -\t{(virginica_df['sepal width'].min()):.1f}\t\t{(virginica_df['sepal width'].max()):.1f}\t\t{(virginica_df['sepal width'].mean()):.1f}\t\t{(virginica_df['sepal width'].median()):.1f}\n\n"
               )

# Create heatmap of correlations
plt.clf()
iriscorr_df = iris_df.drop(columns=['class'])
corr = iriscorr_df.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')
plt.savefig("Iris - correlation heatmap")

# Write correlation values to textfile
petal_corr = iris_df['petal length'].corr(iris_df['petal width'])
sepal_corr = iris_df['sepal length'].corr(iris_df['sepal width'])
setpetal_corr = setosa_df['petal length'].corr(setosa_df['petal width'])
setsepal_corr = setosa_df['sepal length'].corr(setosa_df['sepal width'])
verpetal_corr = versicolor_df['petal length'].corr(versicolor_df['petal width'])
versepal_corr = versicolor_df['sepal length'].corr(versicolor_df['sepal width'])
virpetal_corr = virginica_df['petal length'].corr(virginica_df['petal width'])
virsepal_corr = virginica_df['sepal length'].corr(virginica_df['sepal width'])
textfile.write("I have generated the correlation coefficients in my code, so I will also note the values here in my overview text:\n\n")
textfile.write("\tOverall:\n"
               f"\tPetal Length v Width correlation:\t{(petal_corr):>5.2f}\n"
               f"\tSepal Length v Width correlation:\t{(sepal_corr):>5.2f}\n\n"
               "\tSetosa:\n"
               f"\tPetal Length v Width correlation:\t{(setpetal_corr):>5.2f}\n"
               f"\tSepal Length v Width correlation:\t{(setsepal_corr):>5.2f}\n\n"
               "\tVersicolor:\n"
               f"\tPetal Length v Width correlation:\t{(verpetal_corr):>5.2f}\n"
               f"\tSepal Length v Width correlation:\t{(versepal_corr):>5.2f}\n\n"
               "\tVirginica:\n"
               f"\tPetal Length v Width correlation:\t{(virpetal_corr):>5.2f}\n"
               f"\tSepal Length v Width correlation:\t{(virsepal_corr):>5.2f}\n\n"
               )
