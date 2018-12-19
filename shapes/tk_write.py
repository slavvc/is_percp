#! /usr/bin/env python

import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import gen_samples

class Win(tk.Frame):
    def __init__(self):
        super().__init__()

        self.image = Image.new('L', (200,200))
        self.scaled_image = self.image.resize((28,28), resample=Image.BILINEAR)
        self.image_draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(width=200 + 20 + 20 + 28, height=200 + 20 + 20)
        self.canvas.grid(columnspan=3)
        self.clear()
        self.canvas.bind('<B1-Motion>', lambda e:self.mouse_move(e))
        self.canvas.bind('<Button-1>', lambda e:self.mouse_down(e))
        self.bclear = tk.Button(text='clear')
        self.bclear.bind('<Button-1>', lambda e: self.clear())
        self.bclear.grid(column=0, row=1)
        self.bpred = tk.Button(text='predict')
        self.bpred.bind('<Button-1>', lambda e: self.predict())
        self.bpred.grid(column=1, row=1)
        
        self.bload = tk.Button(text='open')
        self.bload.bind('<Button-1>', lambda e: self.load())
        self.bload.grid(column=2, row=1)

        self.predictor = None
        self.x = 0
        self.y = 0
        self.radius = 2

    def load(self):
        fn = tk.filedialog.askopenfilename()
        if fn != ():
            self.image = Image.open(fn).resize((200,200))
            self.image_draw = ImageDraw.Draw(self.image)
            self.update()

    def update(self):
        self.canvas.delete(tk.ALL)
        self.scaled_image = self.image.resize((28,28), resample=Image.BILINEAR)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.photo_scaled_image = ImageTk.PhotoImage(self.scaled_image)
        self.canvas.create_image(100,100, image=self.photo_image)
        self.canvas.create_image(200 + 20 + 20 + 14,14, image=self.photo_scaled_image)
        
        inp = self.compute_input()
        ainp = gen_samples.adapt_input(inp)
        for x in range(200):
            self.canvas.create_line(((x, 201),(x, 211+inp[200+x]*10)))
        for y in range(200):
            self.canvas.create_line(((201, y),(211+inp[y]*10, y)))
        for x in range(200):
            self.canvas.create_line(((x, 221),(x, 231+ainp[200+x]*10)))
        for y in range(200):
            self.canvas.create_line(((221, y),(231+ainp[y]*10, y)))
        
        # self.image.show()

    def clear(self):
        # self.canvas.delete(tk.ALL)
        # self.canvas.create_rectangle(0,0,224,224,fill='black')
        self.image_draw.rectangle([0,0,200,200], fill='black')
        self.update()

    def mouse_down(self, event):
        self.x, self.y = event.x, event.y
        r = self.radius
        self.image_draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r],
            fill='white'
        )
        self.update()
    def mouse_move(self, event):
        # self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3,
        #     fill='white', outline='white'
        # )
        r = self.radius
        dx = event.x - self.x
        dy = event.y - self.y
        l = (dx*dx+dy*dy)**.5
        dist = 2.5
        if l > dist:
            nx = dx / l
            ny = dy / l
            for i in range(1, int(l // dist) + 1):
                x = self.x + nx*dist*i
                y = self.y + ny*dist*i
                self.image_draw.ellipse([x-r, y-r, x+r, y+r],
                    fill='white'
                )
        self.x, self.y = event.x, event.y
        self.image_draw.ellipse([event.x-r, event.y-r, event.x+r, event.y+r],
            fill='white'
        )
        self.update()

    def compute_input(self):
#        img = self.image.resize((28,28), resample=Image.BILINEAR)
        imarr = np.array(self.image)
        ys = imarr.sum(axis=0).flatten() / 200 / 255 * 2 - 1
        xs = imarr.sum(axis=1).flatten() / 200 / 255 * 2 - 1
        return np.concatenate((xs, ys))
            

    def predict(self):
        inp = self.compute_input()
        inp = gen_samples.adapt_input(inp)
        if self.predictor:
            y = self.predictor.react(inp)
            s = gen_samples.categories[y.argmax()]
            self.canvas.create_text(50,20, text=s, fill='red', font=20)
        else:
            print(inp)

def main():
    root = tk.Tk()
    w = Win()
    # root.geometry('224x224')
    root.mainloop()

if __name__ == '__main__':
    main()
