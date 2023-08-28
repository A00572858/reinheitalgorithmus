import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

# df_raw = pd.read_csv("./Evidencia1/beers.csv")
# df_clean = df_raw[['abv','ibu','style']].dropna().reset_index().copy()

# df_x = df_clean[['abv','ibu']]
# df_y = df_clean[['style']]

# df_y.loc[df_y['style'].str.contains('IPA'), 'style'] = "IPA"

# df_y_t = df_y.copy()

# for i in range(len(df_y_t)):
#     if df_y_t['style'][i] == "IPA":
#         df_y_t['style'][i] = 1
#     else:
#         df_y_t['style'][i] = 0

# print(df_x.head())
# print(df_y_t.head())

# df_x['class'] = df_y_t['style']
# print(df_x.info())

# df_train, df_test = train_test_split(df_x, test_size = 0.2)

# print((len(df_train[df_train['class'] == 1]))/(len(df_train)))
# print((len(df_test[df_test['class'] == 1]))/(len(df_test)))

# df_train_shuffled = df_train.sample(frac=1).reset_index()
# df_test_shuffled = df_test.sample(frac=1).reset_index()

# x_train = df_train_shuffled[['ibu','abv']]
# y_train = df_train_shuffled[['class']]

# print(x_train)
# print(y_train)

def hypothesis(params, samples):
	acum = 0
	for i in range(len(params)):
		acum += params[i] * samples[i]
	acum = acum * (-1)
	acum = 1 / (1 + math.exp(acum))
	return acum

def GradientD(params, samples, y, alfa):
	aux = list(params)
	for j in range(len(params)):
		acum = 0
		for i in range(len(samples)):
			error = hypothesis(params, samples[i]) - y[i]
			acum += error + samples[i][j]
		aux[j] = params[j] - alfa * (1/len(samples)) * acum
	return aux

# params = [0,0,0]
# samples = x_train.to_numpy()
# y = y_train.to_numpy()
# alfa = .03

params = [0,0,0]
samples = [[20.0,0.040],[15.0,0.035],[50.0,0.070],[18.0,0.038],[70.0,0.068],[60.0,0.080],[22.0,0.049]]
y = [0,0,1,0,1,1,0]
alfa = .03

# print(samples)
# print(y)

for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i] =  [1]+samples[i]
	else:
		samples[i] =  [1,samples[i]]

while True:
	oldparams = list(params)
	params = GradientD(params, samples,y,alfa)	
	if(oldparams == params): 
		print ("Samples:")
		print (samples)
		print ("Final params:")
		print (params)
		break



