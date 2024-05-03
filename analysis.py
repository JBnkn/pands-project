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
               f"Sepal Length - Mean: {round(setosa_df['sepal length'].mean(),1)} Median: {round(setosa_df['sepal length'].median(),1)}\n"
               f"Sepal Width  - Mean: {round(setosa_df['sepal width'].mean(),1)} Median: {round(setosa_df['sepal width'].median(),1)}\n"
               f"Petal Length - Mean: {round(setosa_df['petal length'].mean(),1)} Median: {round(setosa_df['petal length'].median(),1)}\n"
               f"Petal Width  - Mean: {round(setosa_df['petal width'].mean(),1)} Median: {round(setosa_df['petal width'].median(),1)}\n\n"
               "Versicolor:\n"
               f"Sepal Length - Mean: {round(versicolor_df['sepal length'].mean(),1)} Median: {round(versicolor_df['sepal length'].median(),1)}\n"
               f"Sepal Width  - Mean: {round(versicolor_df['sepal width'].mean(),1)} Median: {round(versicolor_df['sepal width'].median(),1)}\n"
               f"Petal Length - Mean: {round(versicolor_df['petal length'].mean(),1)} Median: {round(versicolor_df['petal length'].median(),1)}\n"
               f"Petal Width  - Mean: {round(versicolor_df['petal width'].mean(),1)} Median: {round(versicolor_df['petal width'].median(),1)}\n\n"
               "Virginica:\n"
               f"Sepal Length - Mean: {round(virginica_df['sepal length'].mean(),1)} Median: {round(virginica_df['sepal length'].median(),1)}\n"
               f"Sepal Width  - Mean: {round(virginica_df['sepal width'].mean(),1)} Median: {round(virginica_df['sepal width'].median(),1)}\n"
               f"Petal Length - Mean: {round(virginica_df['petal length'].mean(),1)} Median: {round(virginica_df['petal length'].median(),1)}\n"
               f"Petal Width  - Mean: {round(virginica_df['petal width'].mean(),1)} Median: {round(virginica_df['petal width'].median(),1)}\n\n"
               )

# Looking at correlations in the overall dataset, and then on a class-by-class basis
# Create a figure with subplots
plt.clf()
fig, axs = plt.subplots(4, 2, figsize=(12, 24))

# Iris Sepal subplot
x_irisseplen = iris_df['sepal length']
y_irissepwid = iris_df['sepal width']
axs[0,0].scatter(x_irisseplen, y_irissepwid)
m_irissep, b_irissep = np.polyfit(x_irisseplen, y_irissepwid, 1)
axs[0,0].plot(x_irisseplen, m_irissep * x_irisseplen + b_irissep, color='red', label='Trendline')
correlation_coefficient_irissep = np.corrcoef(x_irisseplen, y_irissepwid)[0, 1]
correlation_label_irissep = str(f"Correlation: {round(correlation_coefficient_irissep,2)}")
axs[0,0].legend([correlation_label_irissep, 'Trendline'])
axs[0,0].set_xlabel('Sepal Length', fontsize=10)
axs[0,0].set_ylabel('Sepal Width', fontsize=10)
axs[0,0].set_title('Overall Iris Dataset')

# Iris Petal subplot
x_irispetlen = iris_df['petal length']
y_irispetwid = iris_df['petal width']
axs[0,1].scatter(x_irispetlen, y_irispetwid)
m_irispet, b_irispet = np.polyfit(x_irispetlen, y_irispetwid, 1)
axs[0,1].plot(x_irispetlen, m_irispet * x_irispetlen + b_irispet, color='red', label='Trendline')
correlation_coefficient_irispet = np.corrcoef(x_irispetlen, y_irispetwid)[0, 1]
correlation_label_irispet = str(f"Correlation: {round(correlation_coefficient_irispet,2)}")
axs[0,1].legend([correlation_label_irispet, 'Trendline'])
axs[0,1].set_xlabel('Petal Length', fontsize=10)
axs[0,1].set_ylabel('Petal Width', fontsize=10)
axs[0,1].set_title('Overall Iris Dataset')

# Setosa Sepal subplot
x_setseplen = setosa_df['sepal length']
y_setsepwid = setosa_df['sepal width']
axs[1,0].scatter(x_setseplen, y_setsepwid)
m_setsep, b_setsep = np.polyfit(x_setseplen, y_setsepwid, 1)
axs[1,0].plot(x_setseplen, m_setsep * x_setseplen + b_setsep, color='red', label='Trendline')
correlation_coefficient_setsep = np.corrcoef(x_setseplen, y_setsepwid)[0, 1]
correlation_label_setsep = str(f"Correlation: {round(correlation_coefficient_setsep,2)}")
axs[1,0].legend([correlation_label_setsep, 'Trendline'])
axs[1,0].set_xlabel('Sepal Length', fontsize=10)
axs[1,0].set_ylabel('Sepal Width', fontsize=10)
axs[1,0].set_title('Setosa')

# Setosa Petal subplot
x_setpetlen = setosa_df['petal length']
y_setpetwid = setosa_df['petal width']
axs[1,1].scatter(x_setpetlen, y_setpetwid)
m_setpet, b_setpet = np.polyfit(x_setpetlen, y_setpetwid, 1)
axs[1,1].plot(x_setpetlen, m_setpet * x_setpetlen + b_setpet, color='red', label='Trendline')
correlation_coefficient_setpet = np.corrcoef(x_setpetlen, y_setpetwid)[0, 1]
correlation_label_setpet = str(f"Correlation: {round(correlation_coefficient_setpet,2)}")
axs[1,1].legend([correlation_label_setpet, 'Trendline'])
axs[1,1].set_xlabel('Petal Length', fontsize=10)
axs[1,1].set_ylabel('Petal Width', fontsize=10)
axs[1,1].set_title('Setosa')

# Versicolor Sepal subplot
x_verseplen = versicolor_df['sepal length']
y_versepwid = versicolor_df['sepal width']
axs[2,0].scatter(x_verseplen, y_versepwid)
m_versep, b_versep = np.polyfit(x_verseplen, y_versepwid, 1)
axs[2,0].plot(x_verseplen, m_versep * x_verseplen + b_versep, color='red', label='Trendline')
correlation_coefficient_versep = np.corrcoef(x_verseplen, y_versepwid)[0, 1]
correlation_label_versep = str(f"Correlation: {round(correlation_coefficient_versep,2)}")
axs[2,0].legend([correlation_label_versep, 'Trendline'])
axs[2,0].set_xlabel('Sepal Length', fontsize=10)
axs[2,0].set_ylabel('Sepal Width', fontsize=10)
axs[2,0].set_title('Versicolor')

# Versicolor Petal subplot
x_verpetlen = versicolor_df['petal length']
y_verpetwid = versicolor_df['petal width']
axs[2,1].scatter(x_verpetlen, y_verpetwid)
m_verpet, b_verpet = np.polyfit(x_verpetlen, y_verpetwid, 1)
axs[2,1].plot(x_verpetlen, m_verpet * x_verpetlen + b_verpet, color='red', label='Trendline')
correlation_coefficient_verpet = np.corrcoef(x_verpetlen, y_verpetwid)[0, 1]
correlation_label_verpet = str(f"Correlation: {round(correlation_coefficient_verpet,2)}")
axs[2,1].legend([correlation_label_verpet, 'Trendline'])
axs[2,1].set_xlabel('Petal Length', fontsize=10)
axs[2,1].set_ylabel('Petal Width', fontsize=10)
axs[2,1].set_title('Versicolor')

# Virginica Sepal subplot
x_virseplen = virginica_df['sepal length']
y_virsepwid = virginica_df['sepal width']
axs[3,0].scatter(x_virseplen, y_virsepwid)
m_virsep, b_virsep = np.polyfit(x_virseplen, y_virsepwid, 1)
axs[3,0].plot(x_virseplen, m_virsep * x_virseplen + b_virsep, color='red', label='Trendline')
correlation_coefficient_virsep = np.corrcoef(x_virseplen, y_virsepwid)[0, 1]
correlation_label_virsep = str(f"Correlation: {round(correlation_coefficient_virsep,2)}")
axs[3,0].legend([correlation_label_virsep, 'Trendline'])
axs[3,0].set_xlabel('Sepal Length', fontsize=10)
axs[3,0].set_ylabel('Sepal Width', fontsize=10)
axs[3,0].set_title('Virginica')

# Virginica Petal subplot
x_virpetlen = virginica_df['petal length']
y_virpetwid = virginica_df['petal width']
axs[3,1].scatter(x_virpetlen, y_virpetwid)
m_virpet, b_virpet = np.polyfit(x_virpetlen, y_virpetwid, 1)
axs[3,1].plot(x_virpetlen, m_virpet * x_virpetlen + b_virpet, color='red', label='Trendline')
correlation_coefficient_virpet = np.corrcoef(x_virpetlen, y_virpetwid)[0, 1]
correlation_label_virpet = str(f"Correlation: {round(correlation_coefficient_virpet,2)}")
axs[3,1].legend([correlation_label_virpet, 'Trendline'])
axs[3,1].set_xlabel('Petal Length', fontsize=10)
axs[3,1].set_ylabel('Petal Width', fontsize=10)
axs[3,1].set_title('Virginica')

plt.savefig("Iris - correlation coefficients")

# Write correlation values to textfile
textfile.write("I have generated the correlation coefficients in my code, so I will also note the values here in my overview text:\n\n")
textfile.write("Overall:\n"
               f"Sepal Length v Width correlation: {round(correlation_coefficient_irissep,2)}\n"
               f"Petal Length v Width correlation: {round(correlation_coefficient_irispet,2)}\n\n"
               "Setosa:\n"
               f"Sepal Length v Width correlation: {round(correlation_coefficient_setsep,2)}\n"
               f"Petal Length v Width correlation: {round(correlation_coefficient_setpet,2)}\n\n"
               "Versicolor:\n"
               f"Sepal Length v Width correlation: {round(correlation_coefficient_versep,2)}\n"
               f"Petal Length v Width correlation: {round(correlation_coefficient_verpet,2)}\n\n"
               "Virginica:\n"
               f"Sepal Length v Width correlation: {round(correlation_coefficient_virsep,2)}\n"
               f"Petal Length v Width correlation: {round(correlation_coefficient_virpet,2)}\n\n"
               )