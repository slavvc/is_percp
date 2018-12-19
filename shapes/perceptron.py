import numpy as np
import scipy.special
import json

class PerceptronLayer():
    #N - neurons, M - input
    def __init__(self, N, M, a=10):
        if M < 1:
            raise 'M must be >= 1'
        if N < 1:
            raise 'N must be >= 1'
        self.N = N
        self.M = M
        self.w = np.zeros((N, M+1))
        self.prev_delta_w = np.zeros((N,M+1))
        self.a = a
        self.last_output = np.zeros(N)
#         self.last_output[-1] = 1
        self.delta = np.zeros(N)
#         self.err = 0
    def react(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape != (self.M,):
            raise 'wrong input size'
        self.last_output = scipy.special.expit(self.w.dot(np.append(X, 1))*self.a)
        return self.last_output
    def randomize(self):
        self.w = np.random.random((self.N, self.M+1))*2-1

class Perceptron():
    def __init__(self, Ns, M, a=10):
        if M < 1:
            raise 'M must be >= 1'
        self.M = M
        self.Ns = Ns
        self.a = a
        self.error = 0
        self.layers = []
        for n in Ns:
            self.layers.append(PerceptronLayer(n, M, a))
            M = n
    def react(self, X):
        for l in self.layers:
            X = l.react(X)
#             print(X)
        return X
    def randomize(self):
        for l in self.layers:
            l.randomize()
    def fit(self, Xs, Ys, steps, eta, inertia):
        print('fit begin')
        for step in range(steps):
            for i, X in enumerate(Xs):
                Y = Ys[i]
                y = self.react(X)

                self.layers[-1].delta = self.a*y*(1-y)*(y-Y)

#                 if step % 100 == 0:
#                     print('y: {} -> {}'.format(y[0],Y[0]))
#                     print('neuron delta:', self.layers[-1].delta)

                for j in reversed(range(len(self.layers)-1)):
                    layer = self.layers[j]
                    next_layer = self.layers[j+1]
                    for k in range(layer.N):
                        o = layer.last_output[k]
                        delta = self.a*o*(1-o)
                        delta *= next_layer.w[:,k].dot(next_layer.delta)
                        layer.delta[k] = delta

                delta = -eta*np.outer(self.layers[0].delta,np.append(X, 1))
#                 print('w delta:', delta)
                delta = inertia*self.layers[0].prev_delta_w + (1-inertia)*delta
#                 print('w with inertia:', delta)
                self.layers[0].prev_delta_w = delta
                self.layers[0].w += delta
#                 print('new w:', self.layers[0].w)
                for j in range(1, len(self.layers)):
                    layer = self.layers[j]
                    prev_layer = self.layers[j-1]
                    delta = -eta*np.outer(layer.delta, np.append(prev_layer.last_output, 1))
                    delta = inertia*layer.prev_delta_w + (1-inertia)*delta
                    layer.prev_delta_w = delta
                    layer.w += delta
#             y=self.react(Xs[0])
#             print(y)
        print('fit end')
        self.error = 0
        for i, X in enumerate(Xs):
            Y = Ys[i]
            y = self.react(X)
            self.error += sum(0.5*(Y-y)**2)

def store(p, fn):
    res = [p.a, p.M] + [x.w.tolist() for x in p.layers]
    with open(fn, 'wt') as f:
        json.dump(res, f)

def load(fn):
    with open(fn, 'rt') as f:
        res = json.load(f)
    a = res[0]
    M = res[1]
    Ns = [len(x) for x in res[2:]]
    p = Perceptron(Ns, M, a)
    for i, x in enumerate(res[2:]):
        p.layers[i].w = np.array(x)
    return p