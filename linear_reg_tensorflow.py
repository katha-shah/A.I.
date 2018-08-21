import tensorflow as tf
import numpy as np

#hyper-parameters
num_epochs=1000
learning_rate = 0.01

x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

theta0 = tf.Variable(tf.random_normal(shape=[1,1]))
theta1 = tf.Variable(tf.random_normal(shape=[1,1]))

def get_data():
    """
        return: data x,y
    """
    xtrain = np.arange(2,19,3,dtype=np.float32)
    #y = m * x + c
    #y = [ 2*xi + 1 for xi in x]
    ytrain = 5 * xtrain + 2.5
    return xtrain , ytrain

def lin_reg_model(x,theta0,theta1):
    #shape of X = [1,n_samples]
    #shape of theta1 = [1,1]
    #shape of theta0 = [1,1]

    ypred = tf.add(tf.matmul(theta1,x),theta0)
    return ypred

xtrain , ytrain = get_data()

xtrain = xtrain.reshape([1, -1])
ytrain = ytrain.reshape([1, -1])
n_samples = xtrain.shape[1]

ypred = lin_reg_model(xtrain,theta0,theta1)
cost = tf.reduce_sum(tf.square(ypred - y )/(2*n_samples))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1,num_epochs):
        sess.run(optimizer,feed_dict={x:xtrain,y:ytrain})

        if epoch%100 == 0 or epoch == 1:
            cost_,t0,t1 = sess.run([cost,theta0,theta1],feed_dict={x:xtrain,y:ytrain})
            print("epoch ={} | cost = {} | theta0 = {} theta1 = {}".format(epoch,cost_,t0,t1))
