#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from itertools import combinations

import classifier

#classifier.load('e.json')
classifier.model = classifier.keras.models.load_model('g.hdf5')

size = (360, 270)
fps = 10
confidence_threshold = 0.9
square_deviation = 0.1
num_of_best = 9

class Win(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.cap = None
        self.last_capture = None
        self.spf = None
        self.best = []
        self.tkbest = []
        
        
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
        self.canvas = tk.Canvas(width=3*size[0]+2*28, height=max(size[1], num_of_best*28))
        self.canvas.grid(columnspan=2)

        
        tk.Button(text='camera', command=lambda: self.camera())\
        .grid(column=0, row=1)
        tk.Button(text='shot', command=lambda: self.shot())\
        .grid(column=1, row=1)
        
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection()
        
        self.pimage = Image.fromarray(self.image)
        self.tkimage = ImageTk.PhotoImage(self.pimage)
        self.canvas.create_image(size[0]//2, size[1]//2, image=self.tkimage)
        
    def __enter__(self):
        return self
    def __exit__(self, ex_type, ex_val, traceback):
        if self.cap is not None:
            self.cap.release()
    
    def update_best(self, conf, num, pic):
        i = -1
        for j in range(len(self.best)):
            if self.best[j][0] <= conf:
                i = j
                break
        if i > -1 or len(self.best) < num_of_best:
            p = Image.fromarray(pic)
            tkp = ImageTk.PhotoImage(p)
            if i == -1:
                i = len(self.best)
            self.best.insert(i, (conf, num))
            self.tkbest.insert(i, (p, tkp))
            
    def clear_best(self):
        self.best.clear()
        self.tkbest.clear()
            
    def update_image(self):
        self.clear_best()
        
        self.image = cv2.GaussianBlur(self.image, (5,5), 0)
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
#        gray = cv2.fastNlMeansDenoising(gray)
        
        gray = cv2.Canny(gray, 50, 100)
        
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
#        rects = self.ss.process()
        rects = self.contourboxes(contours)
        
#        edges = self.edge_detection.detectEdges(np.float32(self.image) / 255.0)
#    
#        orimap = self.edge_detection.computeOrientation(edges)
#        edges = self.edge_detection.edgesNms(edges, orimap)
#    
#        edge_boxes = cv2.ximgproc.createEdgeBoxes()
#        edge_boxes.setMaxBoxes(30)
#        rects = self.edge_boxes.getBoundingBoxes(edges, orimap)
        
        
        
        
        
        iw, ih = gray.shape
        n = 0
        for i in range(min(len(rects), 100)):
            r = rects[i]
            x, y, w, h = r
            k = w/h
            if k > 1 + square_deviation or k < 1 - square_deviation:
                r = self.adjust_rect(r)
                if r is None:
                    continue
            x, y, w, h = r
#            if x < 0 or y < 0 or x+w >= iw or y+h >= ih:
#                continue
#            print(r, gray.shape)
            patch = cv2.resize(gray[y:y+h, x:x+w], (28,28))
            ma, mi = patch.max(), patch.min()
            if ma == mi:
                continue
            
            patch = self.process_patch(patch)
            ans = classifier.predict(patch)
            if ans[0] > confidence_threshold:
#                if x+w >= iw or y+28 >= ih:
#                    continue
                self.update_best(ans[0], ans[1], patch)
                
                self.image[y:y+h, x:x+w, 0] = cv2.resize(patch, (w,h))
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0,255,0), 1)
                cv2.putText(
                        self.image, str(ans[1]), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                        )
                n += 1
        
        
        gray2 = cv2.drawContours(gray.copy(), contours, -1, 127, -1)
        
        for r in self.contourboxes(contours):
            cv2.rectangle(gray2, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), 255, 1)
        
        self.pimage = Image.fromarray(self.image)
        self.tkimage = ImageTk.PhotoImage(self.pimage)

        self.pgray = Image.fromarray(gray)
        self.tkgray = ImageTk.PhotoImage(self.pgray)
        self.pgray2 = Image.fromarray(gray2)
        self.tkgray2 = ImageTk.PhotoImage(self.pgray2)
        
        self.canvas.delete(tk.ALL)
        self.canvas.create_image(size[0]//2, size[1]//2, image=self.tkimage)
        
        self.canvas.create_image(
                size[0] + 2*28 + size[0]//2, size[1]//2, image=self.tkgray
                )
        self.canvas.create_image(
                2*size[0] + 2*28 + size[0]//2, size[1]//2, image=self.tkgray2
                )
        
        for i, (conf, num) in enumerate(self.best):
            self.canvas.create_image(
                    size[0]+14, 14 + 28*i, image=self.tkbest[i][1]
                    )
            self.canvas.create_text(
                    size[0]+28+14, 8 + 28*i, text=str(num), fill='red'
                    )
            self.canvas.create_text(
                    size[0]+28+14, 18 + 28*i, text='%.2f'%conf, fill='green'
                    )
        
        if self.spf is not None:
            self.canvas.create_text(
                    25, 10, text='%.2f' % (1/self.spf), fill='red'
                    )
        
        self.canvas.create_text(
                150, 10, text='%d of rects' % len(rects), fill='red'
                )
        self.canvas.create_text(
                220, 10, text='%d shown' % n, fill='red'
                )
    def adjust_rect(self, rect):
        x, y, w, h = rect
        if w < h:
            dw = h - w
            x -= dw // 2
            w += dw
        else:
            dh = w - h
            y -= dh // 2
            h += dh
        if x >= 0 and y >= 0 and x+w < size[0] and y+h < size[1]:
            return (x, y, w, h)
        else:
            return None
    def contourboxes(self, contours):
        s = [x.tolist() for x in contours]
        ss = sum(map(lambda r: [sum(x, []) for x in combinations(s, 1)], range(1, len(s)+1)), [])
        rs = [cv2.boundingRect(np.array(x)) for x in ss]
        return rs
    def process_patch(self, patch):
        patch = (patch - patch.min()) / (patch.max() - patch.min())
        Y,X = np.mgrid[:28, :28]
        s = patch.sum()
        y = (patch * Y).sum() / s
        x = (patch * X).sum() / s
        
        patch = (patch * 255).astype('uint8')
        
        r = np.zeros(patch.shape, dtype='uint8')
        
        k = 1.5
        
        for i in range(28):
            for j in range(28):
                ii = int((i - 14) * k + y)
                jj = int((j - 14) * k + x)
                if ii >= 0 and ii < 28 and jj >= 0 and jj < 28:
                    r[i, j] = patch[ii, jj]
                else:
                    r[i, j] = 0
        
        return r
    
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
            
            
            self.ss.setBaseImage(self.image)
            self.ss.switchToSelectiveSearchFast()
            self.ss.switchToSingleStrategy()
            
            self.update_image()
            self.after_id = self.after(1000//fps, self.capture)
    def shot(self):
        b = self.cap is None
        if b:
            self.cap = cv2.VideoCapture(0)
            
        ret, frame = self.cap.read()
        
        if b:
            self.cap.release()
            self.cap = None
            
        frameim = Image.fromarray(frame).resize(size)
        self.image = np.array(frameim)
        
        self.ss.setBaseImage(self.image)
        self.ss.switchToSelectiveSearchQuality()
        
        self.update_image()

    

if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    
    root = tk.Tk()
    with Win(root) as w:
        tk.mainloop()