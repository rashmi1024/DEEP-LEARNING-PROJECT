# DEEP-LEARNING-PROJECT

COMPANY: CODETECH IT SOLUTION

NAME: RASHMI KUMARI 

INTERN ID:CT04DN978

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

TASK DESCRIPTION :

Overview: This task builds upon the data preprocessing pipeline created in Task 1. The objective is to train a machine learning model that predicts whether an employee will churn (leave the company) or not, based on transformed employee data. Using a Random Forest Classifier, the model is trained and evaluated, and the predictions are saved for future use.

The Random Forest algorithm was chosen due to its reliability, ability to handle categorical and numerical data, and strong performance on classification tasks. This version of the task uses the transformed files generated earlier:

train_features_v2.csv

test_features_v2.csv

train_labels_v2.csv

test_labels_v2.csv


Steps Involved:

Step 1: Loading Transformed Data The script begins by importing the required CSV files using Pandas. These include both the training and testing features and their respective labels. To avoid common file loading errors, a try-except block is used. The labels are accessed from the 'label' column.

Step 2: Initializing the Model A Random Forest Classifier is initialized using Scikit-learn. It uses 100 decision trees and a fixed random state to ensure consistent results every time it is run.

Step 3: Model Training The model is trained using the .fit() method on the training features and labels. During this stage, the classifier learns how different features correlate with the outcome (churn or not churn).

Step 4: Making Predictions After the model has been trained, it is used to make predictions on the test set. The .predict() function is used here to generate binary labels (0 or 1) for each test example.

Step 5: Evaluation The model's performance is assessed using two key metrics:

Accuracy Score: This metric shows how many predictions were correct out of the total.

Classification Report: It provides a detailed breakdown including precision, recall, f1-score, and support for each class (0 or 1).


These metrics help identify how well the model performs on each category and give a balanced view of its strengths and weaknesses.

Step 6: Saving Predictions Predictions are written to a new CSV file called churn_predictions_v2.csv. This file contains two columns: actual and predicted churn labels. This allows easy cross-checking and further analysis of prediction accuracy.

Step 7: Saving the Trained Model The trained Random Forest model is saved using Joblib as churn_model_v2.pkl. This binary file can be reused in future scripts or deployments without having to retrain the model again.

Conclusion: This task demonstrates the end-to-end machine learning workflow: loading data, training a model, evaluating its performance, and saving both predictions and the trained model. It extends the preprocessing pipeline built in Task 1 and showcases practical implementation of classification using scikit-learn.

The variable names, comments, file outputs, and overall code structure were kept unique in this version to reflect an individual submission while achieving the same learning outcome and performance standards required by the internship.
