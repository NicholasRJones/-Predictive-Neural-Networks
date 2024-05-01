from Optimization.Functions import neuralfunction as nf
from Optimization.Algorithm import classy, optisolve as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project 4
# Obtain Data
data = np.array(pd.read_csv(r'../Data/LiverD.csv'))
# Convert entries
for i in range(578):
    if data[i][1] == 'Male':
        data[i][1] = 2 / 3
    else:
        data[i][1] = 1 / 3
    if data[i][10] == 2:
        data[i][10] = 1
    else:
        data[i][10] = 0

big = []
smol = []
# Normalize entries
for i in range(10):
    big.append(max(data[:][i]))
    smol.append(min(data[:][i]))

for j in range(578):
    for i in range(10):
        data[j][i] = 1 / 3 + 1 / 3 * abs((data[j][i] - smol[i]) / (big[i] - smol[i]))


# Parameter class for project
dimensions = [10, 20, 20, 1]
train = 20
start = 100
para = classy.para(0.0001, 0.19, 0, [np.transpose(data[start:start + train, 0:10]), np.transpose(data[start:start + train, 10]), dimensions], 0, 0, 0)
lengthcalc = dimensions * 1
length = 0
for j in range(len(lengthcalc) - 1):
    length = length + lengthcalc[j] * lengthcalc[j + 1]
initialinput = np.random.rand(length)
# Function class to optimize
pr = classy.funct(nf.nueralfunction, 'LBFGS', 'strongwolfe', initialinput, para, 10)
a = op.optimize(pr)

a = np.array(pd.read_csv(r'../Data/project4data')).reshape((1,length))
paraplot = classy.para(0.0001, 0.19, 0, [np.transpose(data[:, 0:10]), np.transpose(data[:, 10]), dimensions], 0, 0, 0)
# diff = nf.nueralfunction(a[0], paraplot, 3).round() - np.transpose(data[:, 10])
diff = nf.nueralfunction(a[0], paraplot, 4).round() - np.transpose(data[:, 10])
# Count instances for bar graph
count = [0, 0, 0]
for j in range(len(diff[0])):
    if diff[0][j] == -1:
        count[0] = count[0] + 1
    if diff[0][j] == 0:
        count[1] = count[1] + 1
    else:
        count[2] = count[2] + 1
bar = {'False Negatives':count[0], 'Accurate':count[1], 'False Positives':count[2]}
keys = list(bar.keys())
values = list(bar.values())
# Plot results
plt.plot(diff[0], marker = 'o', linestyle = 'none', markerfacecolor = 'black', color = 'red')
plt.show()
plt.bar(keys, values, width = .75, color = 'maroon')
plt.show()
