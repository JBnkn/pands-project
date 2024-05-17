# Programming and Scripting project

## Introduction
This repository contains my analysis of the famous [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). I utilised the [dataset found here](https://archive.ics.uci.edu/dataset/53/iris) for the purposes of my analysis. This analysis was carried out as an end-of-module project for the [Programming and Scripting](https://www.gmit.ie/programming-and-scripting) module as part of the [Higher Diploma in Science in Data Analytics at ATU](https://www.gmit.ie/higher-diploma-in-science-in-computing-in-data-analytics):

> This project concerns the well-known Fisher’s Iris data set. You must research the data set and write documentation and code (in Python) to investigate it. An online search for information on the data set will convince you that many people have investigated it previously. You are expected to be able to break this project into several smaller tasks that are easier to solve, and to plug these together after they have been completed. 

## The Dataset
In 1936, British statistician (and mathematician, and biologist, and geneticist, and more) [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) published his paper [_The use of multiple measurements in taxonomic problems_](https://lgross.utk.edu/Math589Fall2020/RAFisher1936measurementsFlowerTaxa.pdf). To gather the data for this paper, Fisher captured measurements of 150 Iris flowers, evenly distributed across three species of the flower (Setosa, Versicolor, and Virginica). For each flower, Fisher noted the petal length, petal width, sepal length, and sepal width. Typically, petals are brightly coloured, surrounding the reproductive structures of flowers, while sepals are more leaf-like, and enclose the bud of the flower.

<img src="https://content.codecademy.com/programs/machine-learning/k-means/iris.svg" alt="A image indicating the petal and sepal for the Versicolor, Setosa, and Virginica species of Iris flower. Credit: https://www.codecademy.com/" width="600">

With this data, Fisher developed a linear discriminant model to distinguish the species from each other. As a clean, complete, and relatively simple dataset, it is a useful sandbox for those of us taking our first steps into the world of Data Science.

## The Data
The data is comprised of four centimeter measurements (petal and sepal, both length and width) stored as Float64, and a class variable, which is an Object and indicates which species of flower the measurements were taken from. There are no NaN values.

## Visualisations
Below are some data visualisations that I generated in the course of my analysis.

![Histogram distributions of the quantitative variables in the Iris dataset](Iris - distribution of quantitative variables.png)

## Modules
I made use of the following modules as part of this project.
- [pandas](https://pandas.pydata.org/)
- [pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
- [seaborn](https://seaborn.pydata.org/tutorial/introduction.html)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## References
- https://www.w3schools.com/python/python_file_write.asp
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
- https://seaborn.pydata.org/generated/seaborn.pairplot.html
- https://seaborn.pydata.org/generated/seaborn.heatmap.html
- https://numpy.org/
- https://scikit-learn.org/stable/
- I occasionally made use of [ChatGPT](https://chat.openai.com/) throughout the course of my projects. It was a useful tool for debugging code that wasn't working for me and I had reached a dead end with the documentation. When I found complex pieces of code elsewhere, it was helpful at explaining the code line-by-line which aided my comprehension of the code.
- I made use of a variety the YouTube videos in the course of my research, including but not limited to [b001](https://www.youtube.com/@b001), [Indently](https://www.youtube.com/@Indently), [Bro Code](https://www.youtube.com/@BroCodez), and [Koolac](https://www.youtube.com/@Koolac)