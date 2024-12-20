# Fake-Real-Job-Posting-Prediction
Objective
The goal of this project is to identify and classify job postings as either legitimate or fraudulent using machine learning techniques. This project aims to provide an automated solution to detect fake job postings, enhancing user trust on job platforms.

Dataset
Source: Kaggle Dataset - Real or Fake Job Posting Prediction
Number of Records: 17,880 job postings
Number of Attributes: 18
Attributes include: job_id, title, location, company_profile, description, requirements, benefits, fraudulent, etc.
Target Variable: fraudulent (binary classification: 0 = real, 1 = fake)
Data Preprocessing
Steps taken to clean and preprocess the dataset:

Missing Values Handling: Removed or imputed missing data where applicable.
Text Preprocessing:
Converted text to lowercase.
Tokenized text using nltk.
Removed stopwords and non-informative characters.
Used TfidfVectorizer to convert job descriptions to numerical form.
Feature Selection: Selected the most relevant features (description, company_profile, requirements, etc.) for training.
Model Training
We used classification algorithms to predict whether a job posting is fake or real. The key models we applied include:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Decision Tree (tuned with GridSearchCV)
Results
The Random Forest model achieved high accuracy in identifying both real and fake job postings. The precision for fraudulent job postings improved significantly after model tuning, with an accuracy rate of 86%.

Conclusion
This model can serve as an efficient tool for job platforms to automatically detect fraudulent postings and protect users from scams. In future work, we plan to explore additional NLP techniques and alternative classification models to improve the accuracy further.
