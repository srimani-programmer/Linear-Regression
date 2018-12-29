import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data/headbrain.csv')

#print(dataset.head())
# Y = mX + C
X = dataset.iloc[:,3]
Y = dataset.iloc[:,2]

'''
print(X)
print(Y)
'''
'''
plt.scatter(X,Y)
plt.show()
'''
# Building the Model

avg_x = np.mean(X)
avg_y = np.mean(Y)

'''
to find the value of m

m = summation((x - avg_x)(y - avg_y))/summation((x - avg_x)(x - avg_x))

'''

numerator = 0
denominator = 0

for i in range(len(X)):
	numerator += ((X[i] - avg_x)*(Y[i] - avg_y))
	denominator += ((X[i] - avg_x) ** 2)


m = numerator/denominator

'''
 to find the value of C

 Y = mX + C
 C = Y - mX
'''

C = avg_y - (m * avg_x)
'''
print('Slope is:',m)
print('Constant is:',C)
'''
# Making the predictions

Y_predicted = m * X + C

plt.scatter(X,Y)
plt.plot([min(X),max(X)], [min(Y_predicted), max(Y_predicted)], color='red')
plt.title("Predicting the Head Size using Least Squares with Linear Regression")
plt.xlabel('Independent Variable(Brain Size)')
plt.ylabel('Dependent Variable (Head Weight)')
plt.show()



