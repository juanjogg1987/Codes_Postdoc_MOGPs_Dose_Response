import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def lm_rsqr(y_true,y_pred_MOGP):
    y_true = y_true.reshape(-1,1)
    y_pred_MOGP = y_pred_MOGP.reshape(-1,1)
    reg = LinearRegression().fit(y_pred_MOGP, y_true)   #Here fit(X,y) i.e., fit(ind_var, dep_var)
    y_pred = reg.predict(y_pred_MOGP)
    return r2_score(y_true,y_pred), reg

y_pred_MOGP = np.array([0.9,1.5,0.3,0.5,0.23,1.5,0.42])
y_true = np.array([0.8,0.95,0.23,1.5,0.15,0.95,0.34])

r2_metric, model = lm_rsqr(y_true,y_pred_MOGP)

plt.scatter(y_true,y_pred_MOGP)
plt.scatter(y_pred_MOGP.reshape(-1,1),model.predict(y_pred_MOGP.reshape(-1,1)))