import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

'''
    Rodrigo Muñoz Guerrero (A00572858)
    Created: 28/08/2023
    Last edited: 12/09/2023

    Title: Reinheit Algorithmus

    Context:
        This file contains the generation of a classification prediction model made from
		scratch, using the Gradient Descent for optimization finding the local minimum.

        The data analysis is made with real data taken from https://www.kaggle.com/datasets/nickhould/craft-cans
        which was adapted to fit the algorithm and the purpose of it.
'''

'''
    Load data from csv file in repository into a pandas data frame.
'''
df_raw = pd.read_csv("beers.csv")

'''
    The data cleaning starts taking only in consideration the abv (alcohol by volume),
    ibu (international bitterness unit) and its style since these are the only relevant
    variables in the dataset for the algorithm.
    We also drop the data that does not
    have this information because it is crucial for the model to have them all.
'''
df_clean = df_raw[['abv','ibu','style']].dropna().reset_index().copy()

df_x = df_clean[['abv','ibu']]
df_y = df_clean[['style']]

'''
    We then transform all the registers that contain the name "IPA" in its style, we do this to
    include every variation of the style (like session IPA, imperial IPA, etc.)
'''
df_y.loc[df_y['style'].str.contains('IPA'), 'style'] = "IPA"
df_y_t = df_y.copy()

'''
    The same data is transformed into two possible classes, IPA or not IPA. We do this with
    integer values so it is easier to handle and we assign 0 to non-IPAs and 1 to its counterpart.
'''
for i in range(len(df_y_t)):
    if df_y_t['style'][i] == "IPA":
        df_y_t['style'][i] = 1
    else:
        df_y_t['style'][i] = 0

df_x['class'] = df_y_t['style']

'''
    The sklearn library is used again to split our data into two, the train and the test data.
    It is separated into 80%/20% for train and test respectively. We do this to have a way of
    testing our model with data we already know is legit.
'''
df_train, df_test = train_test_split(df_x, test_size = 0.2)

'''
    And we shuffle both the dataframes so we do not get an unpredicted bias on the model.
'''
df_train_shuffled = df_train.sample(frac=1).reset_index()
df_test_shuffled = df_test.sample(frac=1).reset_index()

'''
    We only need to separate both our train and test data into the x sets (Independent variables)
	and the y sets (The dependent or the label to our logistic model)
'''
x_train = df_train_shuffled[['ibu','abv']]
y_train = df_train_shuffled[['class']]

x_test = df_test_shuffled[['ibu','abv']]
y_test = df_test_shuffled[['class']]

y_train['class']=y_train['class'].astype('int')
y_test['class']=y_test['class'].astype('int')

'''
    We create a function to calculate our hypothesis of the best params and, as it is a classification 
	algorithm, we activate it with the Sigmoid function.
'''
def hypothesis(params, samples):
	acum = 0
	for i in range(len(params)):
		acum += params[i] * samples[i]
	acum = acum * (-1)
	acum = 1 / (1 + math.exp(acum))
	return acum

'''
    This function uses the concept of the optimization algorithm "Gradient Descent" which updates the
	params of the algorithm to reach a local minimum error.
'''
def GradientD(params, samples, y, alfa):
	aux = list(params)
	for j in range(len(params)):
		acum = 0
		for i in range(len(samples)):
			error = hypothesis(params, samples[i]) - y[i]
			acum += error * samples[i][j]
		aux[j] = params[j] - alfa * (1/len(samples)) * acum
	return aux

'''
    I could not use the data I cleaned because of an error I could not solve in the designated time
	and will leave the code commented ready for when a solution comes up.
'''
# params = [0,0,0]
# samples = x_train.to_numpy()
# y = y_train.to_numpy()
# alfa = .03

'''
    Meanwhile, we will use a few handwriten samples and labels
'''
params = [0,0,0]
samples = [[.200,0.040],[.150,0.035],[.500,0.070],[.180,0.038],[.700,0.068],[.600,0.080],[.220,0.049]]
y = [0,0,1,0,1,1,0]
alfa = .03

for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i] =  [1]+samples[i]
	else:
		samples[i] =  [1,samples[i]]

epoch = 0

'''
    We run the optimization of the parameters until they do not change at all or we reach
	a certain number of iterations or "epochs"
'''
while True:
	oldparams = list(params)
	params = GradientD(params, samples,y,alfa)	
	if(oldparams == params or epoch == 100000): 
		print ("Final params:")
		print (params)
		break
	epoch += 1

'''
	TEST

	We finally test the model with two test samples.

	With sampletest1 = [1,.500,0.070] we must aproximate to 1.
	Which means the beer is an IPA

	With sampletest2 = [1,.300,0.050] we must aproximate to 0.
	Which means the beer is not an IPA
'''

sampletest1 = [1,.500,0.070]
sampletest2 = [1,.300,0.050]

result = np.dot(sampletest1, params)
result = 1 / (1 + math.exp(-result))
prediction = None
label = "Error: (Could not predict)"

if result > 0.85:
	prediction = 1
	label = "an IPA"
elif result < 0.15:
	prediction = 0
	label = "not an IPA"

print("=== SAMPLE TEST 1 ===")
print("Predicted class: " + str(prediction))
print("Class prediction probability: " + str(result))
print("This beer is " + label)

result = np.dot(sampletest2, params)
result = 1 / (1 + math.exp(-result))
prediction = None
label = "Error: (Could not predict)"

if result > 0.85:
	prediction = 1
	label = "an IPA"
elif result < 0.15:
	prediction = 0
	label = "not an IPA"

print("=== SAMPLE TEST 2 ===")
print("Predicted class: " + str(prediction))
print("Class prediction probability: " + str(result))
print("This beer is " + label)