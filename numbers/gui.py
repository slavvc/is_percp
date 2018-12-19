#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time

size = (400, 300)
fps = 10

class Win(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.cap = None
        self.last_capture = None
        self.spf = None
        
        self.image = np.zeros((size[1], size[0], 3), dtype='uint8')
        
        cv2.putText(
                self.image, 'n', (size[0]//2-20,size[1]//2+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                )
        cv2.circle(self.image, (size[0]//2,size[1]//2), 10, (255, 0, 0))
        cv2.putText(
                self.image, 'thing', (size[0]//2+10,size[1]//2+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                )
        self.canvas = tk.Canvas(width=size[0], height=size[1])
        self.canvas.grid()
        
        tk.Button(text='camera', command=lambda: self.camera())\
        .grid(column=0, row=1)
        
        
        
        self.update_image()
        
    def __enter__(self):
        return self
    def __exit__(self, ex_type, ex_val, traceback):
        if self.cap is not None:
            self.cap.release()
        
    def update_image(self):
        self.pimage = Image.fromarray(self.image)
        self.tkimage = ImageTk.PhotoImage(self.pimage)

        
        self.canvas.delete(tk.ALL)
        self.canvas.create_image(size[0]//2, size[1]//2, image=self.tkimage)
        if self.spf is not None:
            self.canvas.create_text(25, 10, text='%.2f' % (1/self.spf), fill='red')
        
    def camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap.release()
            self.cap = None
        if self.cap is not None:
            self.after_id = self.after(1000//fps, self.capture)
        else:
            self.after_cancel(self.after_id)
    def capture(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            frameim = Image.fromarray(frame).resize(size)
            self.image = np.array(frameim)
            
            if self.last_capture is not None:
                self.spf = time.time() - self.last_capture
            self.last_capture = time.time()
            
            self.update_image()
            self.after_id = self.after(1000//fps, self.capture)
    

if __name__ == "__main__":
    root = tk.Tk()
    with Win(root) as w:
        tk.mainloop()