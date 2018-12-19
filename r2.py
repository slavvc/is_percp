#! /usr/bin/env python

if __name__ == '__main__':
    import pickle, tk_write
    import tkinter as tk
    import sklearn.neural_network
    import numpy as np

    with open('mlp.pickle', 'rb') as f:
        mlp = pickle.load(f)

    def ans(y):
        print(y)
        r = np.zeros(4)
        r[y] = 1
        return r

    mlp.react = lambda X: ans(mlp.predict(np.array([X])))

    root = tk.Tk()
    win = tk_write.Win()
    win.predictor = mlp
    root.mainloop()
