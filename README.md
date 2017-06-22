# Identity Fraud From Enron Email


## Understanding the Dataset and Question

The purpose of this project is to use machine learning tools to identify Enron employees (Persons of Interest -POI) who have committed fraud based on the public Enron financial and email dataset. Enron is an energy trading company that had the largest case of corporate fraud in US history. POIs are one of 3: indicted, settled without admitting guilt, or testified in exchange of immunity.

> 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]


  * ### Dataset Background

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


  * ### Outliers

When I plotted bonus vs. salary, there was an outlier datapoint representing the 'TOTAL' column. I removed it as it's a spreadsheet quirk.

![scatter plot1](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp1.png)

After removing the outlier, the scatterplot spread is clearer now as it was skewed earlier due to the effect of the outlier.

![scatter plot2](https://github.com/jasminej90/dand5-identity-fraud-from-enron-email/blob/master/img/scatterp2.png)



## Optimize Feature Selection/Engineering


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

  * ### Univariate Feature Selection

In order to decide the best features to use, I utilized an automated feature selection function, i.e. SelectKBest, which selects the K features that are most powerful (where K is a parameter). The function returned the below scores for all the features.

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


algorithm | precision | recall | precision | recall |
algorithm |<td colspan=2> original_features|<td colspan=2> new_features|
|---|---|---|---|---|
Naive Bayes | 0.401150793651 | 0.337827380952 | 0.396896194084 |0.332306547619




  * ### Feature Scaling

Since my selected features had different units and some of the features had very big values, I needed to transform them. I used MinMaxScaler from sklearn to scale all my selected features to a given range (between 0 and 1).

```python
from sklearn import preprocessing

data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
```


## Pick and Tune an Algorithm

> 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I tried 5 different algorithms and ended up using `Naive Bayes` as it scored the highest evaluation metrics. Other algorithms that I tried are:
  - SVM
  - Decision Tree
  - Random Forest
  - Logistic Regression
In general, it took longer to run SVM and Logistic Regression models. Also, I noticed that all algorithms scored high accuracy, which is an indication that accuracy in this case is not the best evaluation metric as one of its shortcoming that it's not ideal for skewed classes (which is the case in our enron dataset).

> 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

Most classifiers come with a set of parameters (with default values) which affect the model, and are passed as arguments to the constructor. Typical examples include `C`, `kernel` and `gamma` for SVM Classifier for example. Tuning the parameters of a classifier (usually referred to as hyperparameters) means to optimize the values of those parameters to enable the algorithm to perform its best (Hyperparameter Optimization). It's a final step in the process of applied machine learning before presenting results. If this step is not done well, it can lead to the model misfitting the data.

  * ### Algorithm Tuning

I used GridSearchCV from sklearn for parameter tuning in the algorithms that had parameters (SVM, Decision Tree, and Logistic Regression). Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.


I created the below function to tune the algorithm using grid search. It prints out the best hyperparameters for the model after performing the tuning for 80 iterations, along with the average evaluation metrics results (accuracy, percision, recall).

```python
def tune_params(grid_search, features, labels, params, iters = 80):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predicts)] 
        pre = pre + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
    print "accuracy: {}".format(np.mean(acc))
    print "precision: {}".format(np.mean(pre))
    print "recall:    {}".format(np.mean(recall))
    
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
```

**Tuning Support Vector Classification**

```python
from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

tune_params(svm_grid_search, features, labels, svm_param)
```
**Result**

kernel = 'rbf', C = 1000, gamma = 0.01


**Tuning Decision Tree Classification**

```python
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

tune_params(dt_grid_search, features, labels, dt_param)
```

**Result**

splitter = 'random', criterion = 'gini'


**Tuning Logistic Regression Classification**


```python
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_param = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 0.01, 0.001, 0.0001]}
lr_grid_search = GridSearchCV(estimator = lr_clf, param_grid = lr_param)

tune_params(lr_grid_search, features, labels, lr_param)
```

**Result**

C = 0.1, tol = 1


## Validate and Evaluate

> 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validation is the process where a trained model is evaluated with a testing dataset. The classic mistake in validation process is not splitting your data to training/testing datasets, which leads to overfitting. I used Cross Validation train_test_split function to split 30% of my dataset as for testing. then I used sklearn.metrics accuracy, precision and recall scores to validate my algorithms.

```python
from sklearn.cross_validation import train_test_split
eatures_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = 42)
	
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy_score(labels_test, predicts)
precision_score(labels_test, predicts)
recall_score(labels_test, predicts)
```

The table below shows the results for the chosen algorithms.

| Algorithm | accuracy | precision | recall |
| ----------| -------- | --------- | ------ |
|Naive Bayes| 0.84659 | 0.395065 | 0.3348065 |
|Support Vector Machines|0.86846|0.15416|0.0602728|
|Decision Tree|0.81960|0.28101|0.300833|
|Random Forest|0.86875|0.38770|0.16695|
|Logistic Regression|0.8752|0.0|0.0|


> 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

Recall: True Positive / (True Positive + False Negative). Out of all the items that are truly positive, how many were correctly classified as positive. Or simply, how many positive items were 'recalled' from the dataset.

Precision: True Positive / (True Positive + False Positive). Out of all the items labeled as positive, how many truly belong to the positive class.

The chosen algorithm is `Naive Bayes`, which resulted in `precision of 0.395065` and `recall of 0.3348065`


---

References:

- [1] https://www.programiz.com/python-programming/methods/built-in/sorted
- [2] https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work
- [3] http://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
- [4] https://en.wikipedia.org/wiki/Hyperparameter_optimization
- [5] https://en.wikipedia.org/wiki/Cross-validation_(statistics)

