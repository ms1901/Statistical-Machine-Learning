import numpy as np
import matplotlib.pyplot as plt
N=100
#===========================Q1=======================
'''
Generate 200 multivariate (dimension = 2) normally distributed samples,100 of those samples (Class 1)
should be generated from Ɲ (μ1= [[0][0]] , Ʃ1= [[1 0][0 1]] ) and the other 100 (Class 2) should be
generated from Ɲ (μ2= [[1.5][1.5]] , Ʃ2= [[0.8 0][0 0.8]] ).
'''
mean_1=[0,0]
cov_1=[[1,0],[0,1]]
mean_2=[1.5,1.5]
cov_2=[[0.8,0],[0,0.8]]
samples_1=np.random.multivariate_normal(mean_1, cov_1, N) #class1
samples_2=np.random.multivariate_normal(mean_2, cov_2, N) #class2
print("Sample 1")
print(samples_1)
print("Sample 2")
print(samples_2)
#===========================Q2=======================
sample1_half=samples_1[0:50] #training set for class1
mean1_half=np.mean(sample1_half,axis=0) #axis=0 specifies across Columns
A_term=sample1_half-mean1_half
A_term_transpose=np.transpose(A_term)
product=np.dot(A_term_transpose,A_term) #Will give a 2x2 matrix
cov_1half=(1/50)*product #covariance for class 1
#===========================Q3=======================
sample2_half=samples_2[0:50] #training set for class2
mean2_half=np.mean(sample2_half,axis=0) #axis=0 specifies across Columns
A_term=sample2_half-mean2_half
A_term_transpose=np.transpose(A_term)
product=np.dot(A_term_transpose,A_term)
cov_2half=(1/50)*product #covariance for class 2

##===========================Q4=======================
#Plot training samples using scatter plot
plt.scatter(sample1_half[:,0],sample1_half[:,1],c='red',label="sample1")
plt.scatter(sample2_half[:,0],sample2_half[:,1],c='green',label="sample2")
plt.legend()
plt.show()
#===========================Q5=======================
'''print(cov_1half==cov_2half) --->returns false
therefore the covariance matrix are arbitrary and we will use quadratic discriminant function
'''
#Implementing Quadratic discriminant analysis function
def G1(x,C1,M1):
	W1=(-1/2)*np.linalg.inv(C1)
	w1_1=np.dot(np.linalg.inv(C1),M1)
	W1_0=(-1/2)*np.dot(np.transpose(M1),w1_1)-(1/2)*np.log(np.linalg.det(C1))+np.log(1/2)
	first_term=np.dot(np.dot(np.transpose(x),W1),x)
	second_term=np.dot(np.transpose(w1_1),x)
	return first_term+second_term+W1_0
def G2(x,C2,M2):
	W2=(-1/2)*np.linalg.inv(C2)
	w2_1=np.dot(np.linalg.inv(C2),M2)
	W2_0=(-1/2)*np.dot(np.transpose(M2),w2_1)-(1/2)*np.log(np.linalg.det(C2))+np.log(1/2)
	first_term=np.dot(np.dot(np.transpose(x),W2),x)
	second_term=np.dot(np.transpose(w2_1),x)
	return first_term+second_term+W2_0
#===========================Q6=======================	
count_1=0
count_2=0

print(sample1_half)
for x in sample1_half:
	if G1(x,cov_1half,mean1_half)>G2(x,cov_2half,mean2_half):
		print(G1(x,cov_1half,mean1_half))
		count_1=count_1+1
for x in sample2_half:
	if G1(x,cov_1half,mean1_half)<G2(x,cov_2half,mean2_half):
		count_2=count_2+1
print("count1:Correctly classified as class 1")
print(count_1)
print("count2:Correctly classified as class 2")
print(count_2)
