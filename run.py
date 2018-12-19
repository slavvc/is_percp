#! /usr/bin/env python

if __name__ == '__main__':
    import perceptron, pickle, tk_write
    import tkinter as tk

    p = perceptron.load('percp_thick.json')

    root = tk.Tk()
    win = tk_write.Win()
    win.predictor = p
    root.mainloop()
