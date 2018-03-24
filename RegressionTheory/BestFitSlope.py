from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

# XS = np.array([1, 2, 3, 4, 5, 6, 1, 4], dtype=np.float64)
# YS = np.array([5, 4, 6, 5, 6, 7, 1, 4], dtype=np.float64)


# noinspection PyPep8Naming,PyShadowingNames
def best_fit_slope_finder(features, labels):
    m = (((mean(XS) * mean(YS)) - mean(XS * YS)) /
         ((mean(XS) ** 2) - (mean(XS * XS))))
    return m


def y_intercept_finder(features, labels):
    b = (mean(YS) - m * mean(XS))
    return b


def squared_error(point, line):
    return sum((line - point) ** 2)


def coefficient_of_determination(point, line):
    global y_mean_line
    y_mean_line = [mean(point) for i in point]
    squared_error_regression = squared_error(point, line)
    squared_error_y_mean = squared_error(point, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)


def create_dataSet(hm, variance, step=2, correlation=False):
    val = 1
    YS = []
    for i in range(hm):
        data = val + random.randrange(-variance, variance)
        YS.append(data)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    XS = [i for i in range(len(YS))]
    return np.array(XS, dtype=np.float64), np.array(YS, dtype=np.float64)


XS, YS = create_dataSet(1000, 500, 7, correlation='pos')

m = best_fit_slope_finder(XS, YS)

b = y_intercept_finder(XS, YS)

# Gets coordinates for best fit line plot
regression_line = []
for i in XS:
    ordinate = (m * i + b)
    regression_line.append(ordinate)

# Predictions
predict_x = np.array([120, 30, 92], dtype=np.float64)
predictions = []
for j in predict_x:
    predict_ordinate = (m * predict_x + b)
    predictions.append(predict_ordinate)

r_squared = coefficient_of_determination(YS, regression_line)

# Output:
print('Slope = ', m, 'Y intercept = ', b, 'Ordinates =', regression_line, 'Predict_Ordinates = ',
      predictions, 'R^2 = ', r_squared, sep='\n')

plt.scatter(XS, YS, color='b', label='data')
plt.scatter(predict_x, predict_ordinate, color='g', label='prediction')
plt.plot(XS, regression_line, label='regression_line')
plt.plot(YS, y_mean_line, label='y_mean_line')
plt.legend(loc=4)
plt.show()
