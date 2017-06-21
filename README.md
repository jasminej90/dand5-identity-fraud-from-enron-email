# Identity Fraud From Enron Email



> 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The purpose of this project is to use machine learning tools to identify Enron employees (Persons of Interest -POI) who have committed fraud based on the public Enron financial and email dataset. Enron is an energy trading company that had the largest case of corporate fraud in US history. POIs are one of 3: indicted, settled without admitting guilt, or testified in exchange of immunity.

##### Dataset Background

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


##### Outliers

When I plotted bonus vs. salary, there was an outlier datapoint representing the 'TOTAL' column. I removed it as it's a spreadsheet quirk.

![scatter plot1](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp1.png)

After removing the outlier, the scatterplot spread is clearer now as it was skewed earlier due to the effect of the outlier.

![scatter plot2](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp2.png)




> 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]


I created two new features which I added to my original features_list and passed it to SelectKBest function for features selection. The two features I created are: `fraction_from_poi`, which represents the ratio of the messages from POI to this person against all the messages sent to this person, and `fraction_to_poi`, ratio from this person to POI against all messages from this person.

This code computes the fraction of messages to/from that person that are from/to a POI, given a number of messages to/from POI (numerator) and number of all messages to/from a person (denominator).

```python
def computeFraction(poi_messages, all_messages):
	fraction = 0
	if poi_messages != 'NaN' and all_messages != 'NaN':
		fraction = poi_messages/float(all_messages)
```

I used the above function then to calculate the fraction for each employee in my dataset, and added the new features to my original dataset.

```python
for emp in my_dataset:
	from_poi_to_this_person = my_dataset[emp]['from_poi_to_this_person']
	to_messages = my_dataset[emp]['to_messages']
	fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
	my_dataset[emp]['fraction_from_poi'] = fraction_from_poi

	from_this_person_to_poi = my_dataset[emp]['from_this_person_to_poi']
	from_messages = my_dataset[emp]['from_messages']
	fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
	my_dataset[emp]['fraction_to_poi'] = fraction_to_poi
```

##### Univariate Feature Selection

In order to decide the best features to use, I utilized an automated feature selection function, i.e. SelectKBest, which returned back the below scores for all the features.

```python
[('exercised_stock_options', 25.097541528735491),
('total_stock_value', 24.467654047526398),
('bonus', 21.060001707536571),
('salary', 18.575703268041785),
('fraction_to_poi', 16.641707070468989),
('deferred_income', 11.595547659730601),
('long_term_incentive', 10.072454529369441),
('restricted_stock', 9.3467007910514877),
('total_payments', 8.8667215371077717),
('shared_receipt_with_poi', 8.7464855321290802),
('loan_advances', 7.2427303965360181),
('expenses', 6.2342011405067401),
('from_poi_to_this_person', 5.3449415231473374),
('other', 4.204970858301416),
('fraction_from_poi', 3.2107619169667441),
('from_this_person_to_poi', 2.4265081272428781),
('director_fees', 2.1076559432760908),
('to_messages', 1.6988243485808501),
('deferral_payments', 0.2170589303395084),
('from_messages', 0.16416449823428736),
('restricted_stock_deferred', 0.06498431172371151)]
```

I decided to take the first 10 features (k = 10) along with POI as they obtained the highest scores from SelectKBest results. You can notice that one of my engineered features recevied a high score and so I included it in my final features list.

```python
# KBest Features
['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary',
'fraction_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock',
'total_payments', 'shared_receipt_with_poi']
```

##### Feature Scaling

Since my selected features had different units and some of the features had very big values, I needed to transform them. I used MinMaxScaler to scale all my selected features to a given range (between 0 and 1).

```python
from sklearn import preprocessing

data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
```


> 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]



> 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]



> 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]



> 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]




---

References:
[1] https://www.programiz.com/python-programming/methods/built-in/sorted
[2] https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work
