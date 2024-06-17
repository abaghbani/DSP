import numpy as np
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor

y = lambda a:2*a+5
x=np.arange(-5,5).reshape(-1,1)
noise1=np.array([-1,0,-2,2,1,-2,2,0,-1,-1,]).reshape(-1,1)
noise2=np.array([-1,0,-2,20,21,-2,2,0,-1,-1,]).reshape(-1,1)

ols1=LinearRegression()
ols1.fit(x,y(x)+noise1)

ols2=LinearRegression()
ols2.fit(x,y(x)+noise2)

theilsen1=TheilSenRegressor(random_state=0)
theilsen1.fit(x,y(x)+noise1)

theilsen2=TheilSenRegressor(random_state=0)
theilsen2.fit(x,y(x)+noise2)

plt.plot(x,y(x)+noise1,"ro",label="DataSet1")
plt.plot(x,y(x)+noise2,"b+",label="DataSet2 with outlier data")

plt.plot(x,x*ols1.coef_+ols1.intercept_,label="OLS DataSet1")
plt.plot(x,x*ols2.coef_+ols2.intercept_,label="OLS DataSet2")

plt.plot(x,x*theilsen1.coef_+theilsen1.intercept_,label="Theilsen DataSet1")
plt.plot(x,x*theilsen2.coef_+theilsen2.intercept_,label="Theilsen DataSet2")

plt.legend()
plt.show()
