## Project Goals & Analysis
This repository explores the history of Egyptian pyramids using Python and common data science libraries. The key analyses include:

Exploratory Data Analysis (EDA): An in-depth look at the distribution of pyramid heights, volumes, and construction trends across different dynasties.

Data Visualization: Rich visualizations are used to show the relationship between pyramid dimensions and their historical timeline, including:

Distribution of pyramid heights.

Pyramid height vs. dynasty.

A correlation heatmap of numerical features.

Regression Modeling: Applied Linear and Polynomial Regression to model the rise and fall of pyramid height over the dynasties, capturing the non-linear trend of this ancient architectural practice.

Decline Prediction: A predictive model was built based on the frequency of construction to forecast the dynasty when pyramid-building activity likely ceased.

Dynasty Classification: A Random Forest Classifier was trained to predict a pyramid's dynasty based on its physical attributes (e.g., height, base, volume), achieving a classification accuracy score.

## Technologies Used
Python

Pandas for data manipulation and analysis.

NumPy for numerical operations.

Matplotlib & Seaborn for data visualization.

Scikit-learn for machine learning models and evaluation.

## Dataset
The project uses the Egyptian Pyramids Dataset available on Kaggle.

## How to Run
To run this project on your local machine, follow these steps:

Clone the repository: git clone https://github.com/your-username/egyptian-pyramids-analysis.git
cd egyptian-pyramids-analysis

Install the required libraries

Run the script:
Ensure the pyramids.csv file is in the correct path and execute the main Python script.
