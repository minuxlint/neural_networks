import numpy as np
import random

class neuronet():

    def __init__(self, L, l_rate = 0.5):        # TODO: add comments
        
        self.L      = L
        self.size   = len(L)
        self.l_rate = l_rate
        
        self.W = [(np.random.rand(self.L[i+1], self.L[i]) - 0.5) for i in range(self.size-1)]       # W - an array of matrixes with weights
        self.B = [(np.random.rand(self.L[i+1]) - 0.5) for i in range(self.size-1)]                  # B - an array of vectors with bias weights

        self.I = [None for i in range(self.size)]                                                   # I - an array of vectors with inputs of the activation functions
        self.O = [None for i in range(self.size)]                                                   # O - an array of vectors with outputs of the activation functions

        self.dW = None
        self.dB = None

    def feed_forward(self, inp):        # TODO: add comments
        self.O[0] = inp
        self.I[1] = self.W[0].dot(inp) + self.B[0]

        for i in range(1, self.size-1):
            self.O[i]   = self.f(self.I[i])
            self.I[i+1] = self.W[i].dot(self.O[i]) + self.B[i]

        self.O[self.size-1] = self.g(self.I[self.size-1])

    def back_propagation(self, inp, target):    # TODO: add comments
        
        self.feed_forward(inp)
        dLdI = self.gLoss_der(self.O[self.size-1], target)                                      # dLdI - a vector of partial derivatives d/dI (Loss)

        for i in range(self.size-2, 0, -1):

            temp = dLdI.reshape(self.L[i+1], 1) @ self.O[i].reshape(1, self.L[i])
            self.dW[i] = temp * self.l_rate                                                     # TODO: add support for batch training
            self.dB[i] = dLdI * self.l_rate

            dLdI = self.f_der(self.I[i]) * (dLdI.dot(self.W[i]))

        temp = dLdI.reshape(self.L[1], 1) @ self.O[0].reshape(1, self.L[0])
        self.dW[0] = temp * self.l_rate
        self.dB[0] = dLdI * self.l_rate

    def train_step(self, inp, target, l_rate=-1):
        if l_rate == -1:
            l_rate = self.l_rate

        self.dW = [np.zeros((self.L[i+1], self.L[i])) for i in range(self.size-1)]
        self.dB = [np.zeros((self.L[i+1],))  for i in range(self.size-1)]
    
        self.back_propagation(inp, target)

        for i in range(self.size-1):
            self.W[i] -= self.dW[i]
            self.B[i] -= self.dB[i]

        return self.Loss(self.O[-1], targets[-1])

#    def train(self, inps, targets, l_rate=-1, batch_size=1):
#        if l_rate == -1:
#            l_rate = self.l_rate
#
#        self.dW = [np.zeros((self.L[i+1], self.L[i])) for i in range(self.size-1)]
#        self.dB = [np.zeros((self.L[i+1],))  for i in range(self.size-1)]
#
#        for i in range(len(inps)):
#            self.back_propagation(inps[i], targets[i])
#
#            for i in range(self.size-1):
#                self.W[i] -= self.dW[i]
#                self.B[i] -= self.dB[i]
#
#        return #self.Loss(self.O[-1], targets[-1])

    def predict(self, inp):

        self.feed_forward(inp)
        return self.O[self.size-1]

    def f(self, x):
        return 1/(1+np.exp(-x))
    def g(self, x):
        temp = np.exp(x)
        return temp/np.sum(temp)
    def Loss(self, y, target):
        return -np.sum(y*np.log(target))
    def gLoss_der(self, y, target):
        return y-target
    def f_der(self, x):
        return self.f(x)*(1-self.f(x))

def scale_input(inp):
    s = np.sum(inp)
    return inp/s





data_file = open("/home/mrrobot/AI/mnist/mnist_train_100.csv")
train_data = np.asfarray([line.split(",") for line in data_file.readlines()])
data_file.close()
for i in range(len(train_data)):
    train_data[i][1:] = scale_input(train_data[i][1:])

data_file = open("/home/mrrobot/AI/mnist/mnist_test_10.csv")
test_data = np.asfarray([line.split(",") for line in data_file.readlines()])
data_file.close()
for i in range(len(test_data)):
    test_data[i][1:] = scale_input(test_data[i][1:])

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 300
net = neuronet([input_nodes, hidden_nodes, output_nodes], learning_rate)

for epoch in range(epochs):
    np.random.shuffle(train_data)
    for record in train_data:
        inputs = record[1:]
        targets = [0] * output_nodes
        targets[int(record[0])] = 1
        net.train_step(inputs, targets)
        
    log_train = []
    log_test = []
    for record in train_data:
        inputs = record[1:]
        answer = np.argmax(net.predict(inputs))
        log_train.append(int(record[0]) == answer)
    for record in test_data:
        inputs = record[1:]
        answer = np.argmax(net.predict(inputs))
        log_test.append(int(record[0]) == answer)
    print(sum(log_train) / len(log_train), sum(log_test) / len(log_test))
