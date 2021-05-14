
from numpy.linalg import inv,pinv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import math
# %matplotlib inline

#A PART
X_train = np.random.uniform(0,10,5)

print(X_train)

#
Y_train = X_train*np.exp(X_train)
print(Y_train)



#b1
var=1
l=1
error_sum=0
for test_data in range(5):
  k=[]
  for i in range(5):
    if i!=test_data:
      arr=[]
      for j in range(5):
        if j!=test_data:
          arr.append(var*np.exp(-np.square(X_train[i]-X_train[j])/(2*(l**2))))
      k.append(arr)
    
  k_star=[]
  for i in range(5):
    if i!=test_data:
      k_star.append(var*np.exp(-np.square(X_train[i]-X_train[test_data])/(2*(l**2))))

  y=[]
  for i in range(5):
    if i!=test_data:
      y.append(Y_train[i])
  
  k_star=np.matrix(k_star)
  k_doublestar=var
  y=np.matrix(y)
  mu=k_star*inv(k)*np.transpose(y) #lecture 12
  #print(flatten(mu))
  mean=mu[0][0]
  mean=np.ravel(mean)
  k_doublestar=np.matrix(k_doublestar)
  var_star=k_doublestar+np.dot(np.dot(k_star,np.linalg.inv(np.matrix(k))),np.transpose(k_star))
  new_var=var_star[0][0]
  s=np.random.multivariate_normal(mean, new_var)
  #print(s)
  error_sum=error_sum+(s-Y_train[test_data])**2
print(math.sqrt(error_sum/5))

print(k_star.shape)
print(np.matrix(k).shape)
print(k_doublestar.shape)

#b2
Minimum_mean_error_value=float('inf')
Variance_value_pref=var
l_value_pref=l
var=0
while var<100:
  L=1
  while L<100:
    sum=0
    for test in range(5):
      k=[]
      for i in range(5):
        if i!=test:
          arr=[]
          for j in range(5):
            if j!=test:
              arr.append(var*np.exp(-np.square(X_train[i]-X_train[j])/(2*(L**2))))
          k.append(arr)
      k_star=[]
      for i in range(5):
        if i!=test:
          k_star.append(var*np.exp(-np.square(X_train[i]-X_train[test])/(2*(L**2))))
      
      y=[]
      for i in range(5):
        if i!=test:
          y.append(Y_train[i])

      y=np.matrix(y)
      k_star=np.matrix(k_star)
      k_doublestar=np.matrix(var)
      s=0
      if(np.linalg.det(k)):
        mu=k_star*inv(k)*np.transpose(y)
        mean=mu[0][0]
        mean=np.ravel(mean)
        var_star=k_doublestar+np.dot(np.dot(k_star,np.linalg.inv(np.matrix(k))),np.transpose(k_star))
        new_var=var_star[0][0]
        s=np.random.multivariate_normal(mean, new_var)
        
      
      sum+=(s-Y_train[test])**2

#checking for minimum value
    if np.sqrt(sum/5)<Minimum_mean_error_value: 
      Minimum_mean_error_value=np.sqrt(sum/5)
      Variance_value_pref=var
      l_value_pref=L
    L=L+1
  var=var+1


print(Variance_value_pref)
print(l_value_pref)
print(Minimum_mean_error_value)

#C PART SOLUTION

X_test = np.random.uniform(0,10,50)
Y_test = X_test*np.exp(X_test)
print(Y_test)



s=[]
y_pred=[]

for test in range(50):
  k=[]
  for i in range(5):
    arr=[]
    for j in range(5):
      arr.append(Variance_value_pref*np.exp(-np.square(X_train[i]-X_train[j])/(2*l_value_pref**2)))
    k.append(arr)

  k_star=[]
  for i in range(5):
    k_star.append(Variance_value_pref*np.exp(-np.square(X_train[i]-X_test[test])/(2*l_value_pref**2)))
  
  y=[]
  for i in range(5):
    y.append(Y_train[i])
  
  k_star=np.matrix(k_star)
  y=np.matrix(y)
  mu=k_star*inv(k)*np.transpose(y)
  temp=mu.item(0,0)
  temp=np.ravel(temp)
  k_doublestar=Variance_value_pref
  var_star=k_doublestar+np.dot(np.dot(k_star,np.linalg.inv(np.matrix(k))),np.transpose(k_star))
  s=np.random.multivariate_normal(temp, var_star[0][0])

  y_pred.append(s)
  
plt.scatter(X_test,Y_test, c='#9467bd',label="test")

plt.scatter(X_test,y_pred, c='r',label="pred")
plt.legend(loc='upper left')
plt.show()
