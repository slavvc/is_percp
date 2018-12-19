#! /usr/bin/env python

if __name__ == '__main__':
    import perceptron, pickle, tk_write
    import tkinter as tk
    import sklearn as sk
    import numpy as np

    # Perceptron = perceptron.Perceptron
    # PerceptronLayer = perceptron.PerceptronLayer
    # with open('./w[1,60000x5(repeat),0.1,0.5,random](0.95).pickle', 'rb') as f:
    #     w = pickle.load(f)
    with open('./svc[linear](0.918).pickle', 'rb') as f:
        svc = pickle.load(f)

    def ans(y):
        print(y)
        r = np.zeros(10)
        r[y] = 1
        return r

    svc.react = lambda X: ans(svc.predict(np.array([X])))

    root = tk.Tk()
    win = tk_write.Win()
    win.predictor = svc
    root.mainloop()
