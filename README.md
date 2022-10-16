# English-Language-Proficiency-Evaluator
An English Language Proficiency evaluator built with ML and DL techniques.
The objective is to build a model that can evaluate a english essay with 6 different scores: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.


# The Dataset
The dataset presented here (the ELLIPSE corpus) comprises 3911 argumentative essays written by 8th-12th grade English Language Learners (ELLs).
It includes the full_text of each essay, identified by a unique text_id. The essays are also given a score for each of the six analytic measures above. These analytic measures comprise the target for the competition.



# Data Cleaning and Preprocessing

The dataset was given to many different models so many different preprocessing pipelines were considered.

For ML models the data was tokenized, stripped of punctuation and encoded with TF-IDF algorithm.
For the LSTM neural network, a text vectorization layer  an embedding matrix from spaCy en_core_web_large vocabulary was built 


Lemmatization and stop words removal was considered but not included in the final version because, considering the task, we can get a more precise evaluation about verb tenses and grammar use from those words in their original form.

# The Model

Since we want the model to be able of outputting 5 evaluation scores for each essay, we are dealing with a multi-output regression problem and so we need a wrapper for our ML models (in this case MultiOutputRegressor by Scikit).

Three machine learning models were considered for this task: Linear Regression, Support Vector Machine Regressor, XGBoost for Regression.

Two deep learning models were considered for this task: LSTM Neural Network and BERT Uncased Transformer.

Test size is 20% of the dataset

| Model  | Validation Performance (Mean Absolute Error) | Validation Performance (Mean Squared Error) |
| ------------- | ------------- | ------------- |
| Linear Regression  | 0.5877873583883393  | 0.5478169927430634  |
| SVM Regressor| 0.44030738436414085  | 0.3035907313556448 |
| XGBoost | 0.4459536600999776 | 0.309199442458324 |
| LSTM NN | 0.0141 | 4.6755e-04 |
| BERT | 0.2608  | 0.4026 |
