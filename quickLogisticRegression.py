import numpy as np
from numpy import where, arange, zeros, ones, reshape, array, linspace, logspace, add, dot, transpose, shape, negative
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from scipy.optimize import fmin_bfgs
''' Predict from 2 exam scores whether a student will be admitted to university'''
# ======= load and visualise the dataset =======
data=np.loadtxt('ex2data1.txt',delimiter=',')
print(data[:9,:])

x=data[:,0:2]
y=data[:,-1]
m=len(y)
fig = plt.figure()
positive = where(y==1)
negative = where(y==0)
scatter(x[positive,0],x[positive,1],marker="x",c='red')
scatter(x[negative,0],x[negative,1], marker="o", c='black')
xlabel('Exam 1 Score')
ylabel('Exam 2 Score')
plt.show()

# Sigmoid function
def sigmoid(z):
    return np.divide(1.0,1.0+np.exp(np.negative(z)))

# add column of ones corresponding to x_0=1
X = ones(shape=(m,3))
X[:, 1:3] = x
    
# ======= define the cost function J and gradient function for logistic regression =======
def cost_function(theta, X, y):
        prediction=sigmoid(dot(X,theta))
        J = (1./m)*(dot(transpose(np.log(prediction)),-y)-dot(transpose(np.log(1.-prediction)),(1.0-y)))               
        return J

def grad(theta,X,y):
        prediction=sigmoid(dot(X,theta))
        return 1.0/m*dot(transpose(X),prediction-y)

# ======= Gradient Descent =======
initial_theta = 0.1* np.random.randn(3)
theta_1 = fmin_bfgs(cost_function, initial_theta, fprime=grad, args=(X, y))
print('Theta found by optimization: '+str(theta_1[0])+', '+str(theta_1[1])+', '+str(theta_1[2]))

# ====== Prediction ======
''' Predict outcome from learned model '''
def predict(theta,X):
    return sigmoid(dot(X,theta))>0.5
prob=sigmoid(dot([1, 45, 85],theta_1))
print('For a student with scores 45 and 85, we predict an admission probability of %f\n\n' % (prob))
print('Model accuracy on training set: %f' % ((y[where(predict(theta_1,X) == (y==1))].size / float(y.size)) * 100.0))
