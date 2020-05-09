import numpy as np
import random

class neuronet():

    def __init__(self, L, l_rate = 0.5):        # TODO: add comments
        
        self.L      = L
        self.size   = len(L)
        self.l_rate = l_rate
        
        self.W = [np.random.rand(self.L[i+1], self.L[i]) for i in range(self.size-1)]   # W - an array of matrixes with weights
        self.B = [np.random.rand(self.L[i+1]) for i in range(self.size-1)]              # B - an array of vectors with bias weights

        self.I = [None for i in range(self.size)]                                       # I - an array of vectors with inputs of the activation functions
        self.O = [None for i in range(self.size)]                                       # O - an array of vectors with outputs of the activation functions

        self.dW = [np.zeros((self.L[i+1], self.L[i])) for i in range(self.size-1)]
        self.dB = [np.zeros((self.L[i+1],))  for i in range(self.size-1)]

    def feed_forward(self, inp):        # TODO: add comments

        self.I[0] = inp

        for i in range(self.size-1):
            self.O[i]   = self.f(self.I[i])
            self.I[i+1] = self.W[i].dot(self.O[i]) + self.B[i]

        self.O[self.size-1] = self.g(self.I[self.size-1])

    def back_propagation(self, inp, target):    # TODO: add comments
        
        self.feed_forward(inp)
        dLdI = self.gLoss_der(self.O[self.size-1], target)                                      # dLdI - a vector of partial derivatives d/dI (Loss)

        for i in range(self.size-2, -1, -1):

            temp = dLdI.reshape(self.L[i+1], 1) @ self.O[i].reshape(1, self.L[i])
            self.dW[i] = temp * self.l_rate                                                     # TODO: add support for batch training
            self.dB[i] = dLdI * self.l_rate

            dLdI = self.f_der(self.I[i]) * (dLdI.dot(self.W[i]))

    def train(self, inps, targets, l_rate=-1, batch_size=1):
        if l_rate == -1:
            l_rate = self.l_rate

        for i in range(len(inps)):
            self.back_propagation(inps[i], targets[i])

            for i in range(self.size-1):
                self.W[i] -= self.dW[i]
                self.B[i] -= self.dB[i]

        return #self.Loss(self.O[-1], targets[-1])

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

if __name__ == '__main__':

    train_input = []
    train_target = []
    test_input = []
    test_target = []

    data_file = open("/home/mrrobot/AI/MNIST/mnist/mnist_train_100.csv")
    train_data = np.asfarray([line.split(",") for line in data_file.readlines()])
    data_file.close()
    for i in range(len(train_data)):

        train_input.append(scale_input(train_data[i%3][1:]))
        
        x = [0 for j in range(10)]
        x[int(train_data[i%3][0])] = 1
        train_target.append(np.array(x))

    input_nodes = 784
    hidden_nodes = 150
    output_nodes = 10
    learning_rate = 0.00005
    epochs = 100
    net = neuronet([input_nodes, hidden_nodes,  output_nodes], learning_rate)

    for epoch in range(epochs):
        net.train(train_input, train_target)

    hits = 0
    for i in range(len(train_input)):
        prediction = net.predict(train_input[i])
        prediction = np.argmax(prediction)
        print(prediction)
        if train_target[i][prediction] == 1:
            hits += 1
    print('%d%%' % ((100*hits/len(train_input))))
