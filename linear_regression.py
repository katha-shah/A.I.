import matplotlib.pyplot as plt
from random import random

#parameters
theta0 = random()
theta1 = random()

#hyper-parameters
num_epochs=1000
learning_rate = 0.01


def get_data():
    """
        return: data x,y
    """
    x = list(range(5,20,3))
    y = [ 2*xi + 1 for xi in x]
   # print(" X = {} Y = {}".format(x,y))
    #plt.plot(x,y)
    #plt.show()
    return x,y

def calc_error(y,ypred):

    n_samples = len(y)
    error=(((ypredi-yi)**2) for ypredi,yi in zip(ypred,y))
    error_mean=sum(error)/(2*n_samples)
    return  error_mean

def model(x,y,theta0,theta1):
   # ypred = theta0 + theta1*x
    n_samples = len(y)
    ypred = [theta0 + theta1*xi for xi in x]

    print(theta0)
    print(theta1)
    print(" X = {},  Y = {}, Ypred = {}".format(x,y,ypred))
    error=calc_error(y,ypred)
    print("Error = {}".format(error))

    #gradient-descent
    for epoch in range(1,num_epochs + 1):
       theta0 -= (learning_rate / n_samples) * sum((ypredi - yi for ypredi, yi in zip(ypred, y)))
       theta1 -= (learning_rate / n_samples) * sum(((ypredi - yi) * xi for ypredi, yi, xi in zip(ypred, y, x)))
       ypred = [theta0 + theta1 * xi for xi in x]
       error = calc_error(y, ypred)
       print("EpochError = {} Error={}".format(epoch, error))
      # theta0 -= (learning_rate/n_samples)*(ypred-y)
      # theta1 -= (learning_rate/n_samples)*(ypred-y)*x

    print("theta0 = {} theta1={}".format(theta0,theta1))

    plt.scatter(x,y)
    plt.plot(x,ypred)
    plt.show()

x,y=get_data()

model(x,y,theta0,theta1)