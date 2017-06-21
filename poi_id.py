#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pprint
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'expenses', 'total_stock_value', 'bonus', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
POI_label = ['poi']
total_features = POI_label + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Total number of data points = ', len(data_dict)

# allocation across classes (POI/non-POI)
poi_count = 0
for employee in data_dict:
	if data_dict[employee]['poi'] == True:
		poi_count += 1
print 'number of POI = ', poi_count
print 'number of non-POI = ', len(data_dict) - poi_count

# number of features used
print 'total number of available features for every employee = ', len(total_features), 'which are: ', total_features
print 'number of features used = ', len(features_list), 'which are: ', features_list

# are there features with many missing values? etc.
missing_values = {}
for feat in total_features:
	missing_values[feat] = 0

for emp in data_dict:
	for f in data_dict[emp]:
		if data_dict[emp][f] == 'NaN':
			missing_values[f] += 1
			# fill NaN values
			# data_dict[emp][f] = 0

print 'missing values: ', missing_values

### Task 2: Remove outliers

def show_scatter_plot(dataset, feature1, feature2):
	""" given two features, create a 2D scatter plot
	"""
	data = featureFormat(dataset, [feature1, feature2])
	for p in data:
		x = p[0]
		y = p[1]
		plt.scatter(x, y)

	plt.xlabel(feature1)
	plt.ylabel(feature2)
	plt.show()

# identify outliers
# show_scatter_plot(data_dict, "salary", "bonus")

# remove them
data_dict.pop( "TOTAL", 0 )
# show_scatter_plot(data_dict, "salary", "bonus")


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# create new features
def computeFraction(poi_messages, all_messages):
	""" given a number messages to/from POI (numerator) 
	and number of all messages to/from a person (denominator),
	return the fraction of messages to/from that person
	that are from/to a POI
	"""
	fraction = 0
	if poi_messages != 'NaN' and all_messages != 'NaN':
		fraction = poi_messages/float(all_messages)

   	return fraction

def takeSecond(elem):
	""" take second element for sort
	"""
	return elem[1]

for emp in my_dataset:
	from_poi_to_this_person = my_dataset[emp]['from_poi_to_this_person']
	to_messages = my_dataset[emp]['to_messages']
	fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
	# print fraction_from_poi
	my_dataset[emp]['fraction_from_poi'] = fraction_from_poi

	from_this_person_to_poi = my_dataset[emp]['from_this_person_to_poi']
	from_messages = my_dataset[emp]['from_messages']
	fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
	my_dataset[emp]['fraction_to_poi'] = fraction_to_poi

features_list_n = total_features
features_list_n.remove('email_address')
features_list_n =  features_list_n + ['fraction_from_poi', 'fraction_to_poi']
print features_list_n

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 10)
selector.fit(features, labels)
scores = zip(features_list_n[1:], selector.scores_)
sorted_scores = sorted(scores, key = takeSecond, reverse = True)
print 'SelectKBest scores: ', sorted_scores

kBest_features = POI_label + [(i[0]) for i in sorted_scores[0:10]]
print 'KBest', kBest_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
