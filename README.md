# Identity Fraud From Enron Email



#### 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The purpose of this project is to use machine learning tools to identify Enron employees (Persons of Interest -POI) who have committed fraud based on the public Enron financial and email dataset. Enron is an energy trading company that had the largest case of corporate fraud in US history. POIs are one of 3: indicted, settled without admitting guilt, or testified in exchange of immunity.

> Background on the dataset

The dataset contains about 146 users (18 of them are considered POIs, while 128 are non-POIs) mostly real emails exchanged by senior management of Enron. We can use machine learning on this dataset to answer questions like "can we identify patterns in the emails?", using regression we will be able to understand the relationship between the people's salary and their bonuses for example, and using clustering we can identify who was a member of the board of directors, and who is just an employee.

The dataset has 21 features, that are either financial, email or (POI/non-POI)-related.

**financial features**: [`salary`, `deferral_payments`, `total_payments`, `loan_advances`, `bonus`, `restricted_stock_deferred`, `deferred_income`, `total_stock_value`, `expenses`, `exercised_stock_options`, `other`, `long_term_incentive`, `restricted_stock`, `director_fees`] (all units are in US dollars)

**email features**: [`to_messages`, `email_address`, `from_poi_to_this_person`, `from_messages`, `from_this_person_to_poi`, `shared_receipt_with_poi`] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

**POI label**: [`poi`] (boolean, represented as integer)

Some features have many missing values `NaN`s. Below is a list that shows each feature and how many values it's missing:
```python
{'salary': 51,
'to_messages': 60,
'deferral_payments': 107,
'total_payments': 21,
'loan_advances': 142,
'bonus': 64,
'email_address': 35,
'restricted_stock_deferred': 128,
'total_stock_value': 20,
'shared_receipt_with_poi': 60,
'long_term_incentive': 80,
'exercised_stock_options': 44,
'from_messages': 60,
'other': 53,
'from_poi_to_this_person': 60,
'from_this_person_to_poi': 60,
'poi': 0,
'deferred_income': 97,
'expenses': 51,
'restricted_stock': 36,
'director_fees': 129}
```

A possible fix for the missing values is to replace them with 0s.


> Outliers in the dataset

When I plotted bonus vs. salary, there was an outlier datapoint representing the 'TOTAL' column. I removed it as it's a spreadsheet quirk.

![scatter plot1](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp1.png)

After removing the outlier, the scatterplot spread is clearer now as it was skewed earlier due to the effect of the outlier.

![scatter plot2](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp2.png)




#### 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]



#### 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]



#### 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]



#### 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]



#### 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
