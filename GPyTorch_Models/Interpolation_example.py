# from scipy.interpolate import interp1d
# import numpy as np
#
# x = np.linspace(0, 10, num=11, endpoint=True)
# y = np.cos(-x**2/9.0)
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')
#
# xnew = np.linspace(0, 10, num=41, endpoint=True)
#
# import matplotlib.pyplot as plt
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()

import numpy as np
import pylab
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])*0.11
#xdata = np.array([0.0,   1.0,  3.0, 4.3, 4.8,   5.0,   5.1, 5.3, 6.2])*0.1
ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print (popt)

x = np.linspace(-1, 3, 50)
y = sigmoid(x, *popt)

pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y, label='fit')
pylab.ylim(0, 1.05)
pylab.legend(loc='best')
pylab.show()