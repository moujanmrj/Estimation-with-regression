import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data:
dataSrc = pd.read_csv('./GOOGL.csv', usecols=['Open', 'Date'])
date = dataSrc['Date']
# print(dataSrc)
targetData = np.array(dataSrc['Open'][:len(dataSrc) - 10])

# calculations for linear regression:
# apply least squares ATAx^=ATb :
A = np.array([[1]*len(targetData), np.arange(len(targetData))]).transpose()
AT = np.transpose(A)
ATA = np.matmul(AT, A)
ATb = np.matmul(np.transpose(A), targetData)
x_hat = np.matmul(np.linalg.inv(ATA), ATb)
# print(x_hat)
# note that x_hat[0] is width of origin and x[1] is slope of our line

# make table for linear regression
calculated_value = []
actual_value = []
for i in range(len(dataSrc)-10, len(dataSrc)):
    calculated_value.append(x_hat[1] * i + x_hat[0])
    actual_value.append(dataSrc['Open'][i])
error = np.array(actual_value) - np.array(calculated_value)
data = [calculated_value, actual_value, error]
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None)
# print('Table for linear regression:')
# print(pd.DataFrame(data, ['calculated value:', 'actual value:', 'error:'], date[len(date)-10:]), '\n')

# calculations for 2nd-order regression:
# apply least squares ATAx^=ATb :
A = np.array([[1]*len(targetData), np.arange(len(targetData)), np.arange(len(targetData))**2]).transpose()
ATA = np.matmul(np.transpose(A), A)
ATb = np.matmul(np.transpose(A), targetData)
x_hat = np.matmul(np.linalg.inv(ATA), ATb)

# make table for 2nd-order regression
calculated_value.clear()
for i in range(len(dataSrc)-10, len(dataSrc)):
    calculated_value.append(x_hat[2] * (i ** 2) + x_hat[1] * i + x_hat[0])
error = np.array(actual_value) - np.array(calculated_value)
data = [calculated_value, actual_value, error]
# print('Table for second-degree regression:')
# print(pd.DataFrame(data, ['calculated value:', 'actual value:', 'error:'], date[len(date)-10:]), '\n')

# plotting data source with second-order regression
second_order_regression = x_hat[2] * np.arange(len(date)) ** 2 + x_hat[1] * np.arange(len(date)) + x_hat[0]
x_axis = pd.to_datetime(date)
plt.scatter(x_axis, dataSrc['Open'], s=10, label='actual values')
plt.plot(x_axis, second_order_regression, '-r', label='calculated polynomial')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.title('second order regression')
plt.show()
