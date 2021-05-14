import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

'''
Generate 100 multivariate Gaussian points (dimension=2) with μ= [-1 -1]and Ʃ=[[2 0.5][0.5 1]].Call it X1.
Generate 100 multivariate Gaussian points (dimension=2) with μ= [1 1]and Ʃ=[[2 0.5][0.5 1]].Call it X2.
Z=[X1 X2]
'''
N=100
mean_1=[-1,-1]
cov_1=[[2,0.5],[0.5,1]]
mean_2=[1,1]
cov_2=[[2,0.5],[0.5,1]]
X1=np.random.multivariate_normal(mean_1, cov_1, N)
X2=np.random.multivariate_normal(mean_2, cov_2, N)
Z=np.concatenate((X1,X2),axis=0) #Z
print(len(Z)==200) #n=200--returns true
print(len(Z[0])==2)  #D=dimensions--returns true
#========================Q2========================
'''Plotting X1 and X2 as scatter plot'''
plt.scatter(X1[:,0],X1[:,1],c='red',label="X1")
plt.scatter(X2[:,0],X2[:,1],c='green',label="X2")
plt.legend()
plt.show()
#========================Q3========================

uz=np.mean(Z, axis=0)
print(uz)
X=Z-uz  #Centralised data
uX=np.mean(X,axis=0) 
S=(1/200)*(np.dot(np.transpose(X-uX),X-uX)) #Covariance of X
print(S)
#========================Q4========================
eigenValues, eigenVectors = linalg.eig(S)
print("Eigen_Values")
print(eigenValues)
print("Eigen_Vectors")
print(eigenVectors)
idx = eigenValues.argsort()[::-1]      #non-increasing-order
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
U=eigenVectors
print(U)
#========================Q5=======================
#Projectting X using first column of U
U_first_col_transpose=U[:,0]
Y=(np.dot((U_first_col_transpose),np.transpose(X))) #1 dimnesional-->Projection
#=========================Q6======================
Y1=(np.dot((U_first_col_transpose),np.transpose(X1)))
Y2=(np.dot((U_first_col_transpose),np.transpose(X2)))
plt.scatter(range(1,101),Y1,color="red",label="Y1")
plt.scatter(range(1,101),Y2,color="purple",label="Y2")
plt.legend()
plt.show()
#=========================Q7=======================
#Reconstruction
a = np.array(Y)[np.newaxis]
U=np.array(U_first_col_transpose)[np.newaxis].T
X_reconstructed=np.dot(U,a)
print(len(X_reconstructed)) #200
print(len(X_reconstructed[0])) #2
Z_reconstructed=np.transpose(X_reconstructed)+uz
print(len(Z_reconstructed)==200) #n=200--returns treu
print(len(Z_reconstructed[0])==2)  #D=dimensions--returns true
#print(Z==Z_reconstructed)
sum=0
#Computing MSE without inbuilt methods
for i in range(0,len(Z)):
	for j in range(0,len(Z[0])):
		difference = Z[i][j] - Z_reconstructed[i][j]
		squared_difference = difference**2 
		sum=sum+squared_difference
MSE=sum/(len(Z)*len(Z[0]))
print("MSE")
print(MSE)


