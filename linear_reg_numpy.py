import matplotlib.pyplot as plt
import numpy as np

#parameters
theta0 = np.random.rand()
theta1 = np.random.rand()

#hyper-parameters
num_epochs=1000
learning_rate = 0.01

def get_data():
    """
        return: data x,y
    """
    x = np.arange(2,19,3,dtype=np.float32)
    #y = [ 2*xi + 1 for xi in x]
    y = 5 * x + 2.5
    return x,y

def calc_error(y,ypred):

    n_samples = len(y)
    error = np.square(ypred-y)
    error_mean = np.sum(error)/(2*n_samples)
    return  error_mean

def model(x,y,theta0,theta1):
   # ypred = theta0 + theta1*x
    n_samples = len(y)
    print(theta0)
    print(theta1)
    ypred = theta0 + theta1*x
    error=calc_error(y,ypred)
    print("Error = {}".format(error))

   #gradient descent

    for epoch in range(1,num_epochs+1):
       theta0 -= (learning_rate / n_samples) * np.sum(ypred-y)
       theta1 -= (learning_rate / n_samples) * np.sum((ypred-y)*x)
       ypred = theta0 + theta1 * x
       error = calc_error(y, ypred)
       print("EpochError = {} Error={}".format(epoch, error))
       # theta0 -= (learning_rate/n_samples)*(ypred-y)
       # theta1 -= (learning_rate/n_samples)*(ypred-y)*x
       print("theta0 = {} theta1={}".format(theta0, theta1))



x,y=get_data()

model(x,y,theta0,theta1)