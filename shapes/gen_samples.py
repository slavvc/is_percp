#!/usr/bin/env python3

from PIL import Image

import imageio as iio
import numpy as np
import perceptron2 as perceptron
import os, shutil, time
import cv2

import sklearn.neural_network
import pickle

thickness = 1


def adapt_input(inp):
    threshold = -1
    p1, p2 = inp[:200], inp[200:]
    ip1, jp1 = -1, -1
    for i,x in enumerate(p1):
        if x > threshold:
            ip1 = i
            break
    for i,x in enumerate(reversed(p1)):
        if x > threshold:
            jp1 = 199 - i
            break
    ip2, jp2 = -1, -1
    for i,x in enumerate(p2):
        if x > threshold:
            ip2 = i
            break
    for i,x in enumerate(reversed(p2)):
        if x > threshold:
            jp2 = 199 - i
            break
    pp1 = np.zeros(200)
    for i in range(200):
        k = i / 200
        l = int(ip1 + k * (jp1 - ip1))
        pp1[i] = p1[l]
    pp2 = np.zeros(200)
    for i in range(200):
        k = i / 200
        l = int(ip2 + k * (jp2 - ip2))
        pp2[i] = p2[l]
    return np.concatenate((pp1, pp2))

def compute_input(img):
    imarr = np.array(img)
    ys = imarr.sum(axis=0).flatten() / 200 / 255 * 2 - 1
    xs = imarr.sum(axis=1).flatten() / 200 / 255 * 2 - 1
    return np.concatenate((xs, ys))

def draw_circle(im, brect):
    x, y, dx, dy = brect
    a = min((dx, dy))
    return cv2.circle(im, (int(x+a/2), int(y+a/2)), int(a/2), 255, thickness)

def draw_rect(im, brect):
    x, y, dx, dy = brect
    return cv2.rectangle(im, (x, y), (x+dx, y+dy), 255, thickness)
    
def draw_tri(im, brect):
    x, y, dx, dy = brect
    a = min((dx, dy))
    nx = np.random.randint(x, x+a)
    up = np.random.randint(2)
    
    if up == 1:
        pts = np.int32([(x, y+a), (nx, y), (x+a, y+a)])
    else:
        pts = np.int32([(x, y), (nx, y+a), (x+a, y)])
    return cv2.polylines(im,[ pts], True, 255, thickness)
        
def draw_sin(im, brect):
    x, y, dx, dy = brect
    scale = np.random.random()*3+3
    phase = np.random.random()*2*np.pi
    xs = np.linspace(phase*scale, (2*np.pi+phase)*scale, dx)
    ys = (np.sin(xs)+1)/2*dy+y
    for i in range(dx-1):
        im = cv2.line(im, (x+i, int(ys[i])), (x+i+1, int(ys[i+1])), 255, thickness)
    return im

shapes = [(draw_circle, 'circle'),
          (draw_rect, 'rectangle'),
          (draw_tri, 'triangle'),
          (draw_sin, 'sinusoid')]

categories = [x[1] for x in shapes]

def gen_pics(folder, n):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for i in range(n):
        im = np.zeros((200, 200), dtype='uint8')
        
        dx = np.random.randint(90, 110)
        dy = np.random.randint(90, 110)
        x = np.random.randint(100-dx//2 - 20, 100-dx//2 + 20)
        y = np.random.randint(100-dy//2 - 20, 100-dy//2 + 20)
        shapef, shapen = shapes[np.random.randint(len(shapes))]
        im = shapef(im, (x, y, dx, dy))
#        pim = Image.fromarray(im, mode='L')
        name = os.path.join(folder, '%d_%s.png' % (i, shapen))
#        pim.save()
        iio.imsave(name, im)
        
def makedata(folder):
    if os.path.isdir(folder):
        Xs = []
        Ys = []
        for n in os.listdir(folder):
            fn = os.path.join(folder, n)
            im = Image.open(fn)
            d = compute_input(im)
            
            d = adapt_input(d)
            
            r = categories.index(n.split('_')[1].split('.')[0])
            Xs.append(d)
            Ys.append(r)
        return Xs, Ys
            
def makegif(folder, gif):
    if os.path.isdir(folder):
        arr = []
        for n in os.listdir(folder):
            fn = os.path.join(folder, n)
            im = Image.open(fn)
            imm = np.zeros((240, 240), dtype='uint8')
            imm[:200,:200] = np.array(im, dtype='uint8')
            inp = compute_input(im)
            ainp = adapt_input(inp)
#            print(ainp)
            for x in range(200):
                imm[x, 201:int(211+inp[x]*10)] = 255
            for y in range(200):
                imm[201:int(211+inp[200+y]*10), y] = 255
                
            for x in range(200):
                imm[x, 221:int(231+ainp[x]*10)] = 255
            for y in range(200):
                imm[221:int(231+ainp[200+y]*10), y] = 255
            
            arr.append(imm)
        iio.mimwrite(gif + '.gif', arr, fps=5)
        
def main():
    p = perceptron.Perceptron([50,4], 400, a=0.5)
    p.randomize()
    Xs, Y = makedata('var_fit')
    tXs, tY = makedata('var_test')
    
    Ys = np.zeros((len(Y), 4))
    for i in range(len(Ys)):
        Ys[i][Y[i]] = 1
    tYs = np.zeros((len(tY), 4))
    for i in range(len(tYs)):
        tYs[i][tY[i]] = 1
    
    
    t = time.clock()
    n = 0
    for _ in range(10):
        p.fit(Xs, Ys, 5, 0.1, 0.1)
        n = 0
        for i,X in enumerate(Xs):
            y = p.react(X)
            if Y[i] == np.argmax(y):
                n += 1
        print('train', n / len(Ys))
        n = 0
        for i,X in enumerate(tXs):
            y = p.react(X)
#            print(Y, categories[np.argmax(Y)], categories[tYs[i]])
            if tY[i] == np.argmax(y):
                n += 1
        print('test', n / len(tYs))
        print('time', time.clock() - t)
        
    perceptron.store(p, 'percp3.json')
        
def test(name, folder):
    p = perceptron.load(name)
    Xs, Y = makedata(folder)
    n = 0
    for i,X in enumerate(Xs):
        y = p.react(X)
        if Y[i] == np.argmax(y):
            n += 1
    return n / len(Y)
    
def main2():
    p = sklearn.neural_network.MLPClassifier((100, 100))
    Xs, Ys = makedata('var_fit')
    tXs, tYs = makedata('var_test')
    t = time.clock()
    for _ in range(5):
        p.fit(Xs, Ys)
        n = 0
        for i,X in enumerate(tXs):
            y = p.predict([X])[0]
#            print(y, categories[y], categories[tYs[i]])
            if tYs[i] == y:
                n += 1
        print('test', n / len(tYs))
        n = 0
        for i,X in enumerate(Xs):
            y = p.predict([X])[0]
            if Ys[i] == y:
                n += 1
        print('train', n / len(Ys))
        print('time', time.clock() - t)
    with open('mlp.pickle', 'wb') as f:
        pickle.dump(p, f)