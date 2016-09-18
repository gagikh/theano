#!/usr/bin/python

# Implemnts btree algorithm based on this paper:
# https://pdfs.semanticscholar.org/bef3/27065030de499b52e9a4a1fd8adaeb9b499e.pdf

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy as np
import cv2

import sys

# Some theano functions, later they can be changed to use shared variables.
def get_FuncR():
	x = T.dmatrix("V")
        z = T.dot(x.T, x)
	return theano.function(inputs=[x], outputs=z)

def get_FuncM():
	x = T.matrix("C")
        z = T.sum(x, 0)
	return theano.function(inputs=[x], outputs=z)

def get_FuncN():
	x = T.matrix("C")
        z = x.shape[0]
	return theano.function(inputs=[x], outputs=z)

# Estimate R,M,N params for the cluster C
def estimate_params(FuncR, FuncM, FuncN, I, C):
        v = I[C[:,0], C[:,1]]

        r = FuncR(v)
        m = FuncM(v)
        n = FuncN(v)

        q = m / n

        c = r - np.outer(m, m) / n
        e, v = np.linalg.eig(c)

        i = e.argsort()[::-1]
        i = i[0]
        e = e[i]
        v = v[i,:]
        t = np.dot(v, q)
        return (v, t, e)

# Divide cluster C into two clusters
def divide(I, C, e, t):
        V = I[C[:,0], C[:,1]]
        s = np.dot(V, e)

        i0 = np.nonzero(s <= t)
        i1 = np.nonzero(s >  t)

        C0 = C[i0,:]
        C1 = C[i1,:]

        return (C0[0], C1[0])

# Segment the image
def segment(FuncR, FuncM, FuncN, img, I):
        (H, W, D) = img.shape

        C = np.reshape([[(i,j) for j in range(W)] for i in range(H)], (H*W, 2))

        v, t, e = estimate_params(FuncR, FuncM, FuncN, img, C)

        L = list([C])
        T = list([t])
        E = list([e])
        V = list([v])
        for i in range(I - 1):
                print "cluster E: ", E
                idx = np.argmax(E)

                print "observe cluster: ", idx
                c = L[idx]
                t = T[idx]
                e = E[idx]
                v = V[idx]

                del L[idx]
                del T[idx]
                del E[idx]
                del V[idx]

                C0, C1     = divide(img, c, v, t)
                v0, t0, e0 = estimate_params(FuncR, FuncM, FuncN, img, C0)
                v1, t1, e1 = estimate_params(FuncR, FuncM, FuncN, img, C1)

                L.append(C0);
                L.append(C1);

                E.append(e0);
                E.append(e1);

                T.append(t0);
                T.append(t1);

                V.append(v0);
                V.append(v1);

        return L

# Estimate the mean color of C
def estimate_colors(FuncM, FuncN, I, C):
        V = I[C[:,0], C[:,1]]
        m = FuncM(V)
        n = FuncN(V)
        return m / n

# Set cluster color
def set_color(I, C, color):
        I[C[:,0], C[:,1]] = color

# Run the algorithm
def btquant(FuncR, FuncM, FuncN, img, I):
        L = segment(FuncR, FuncM, FuncN, img, I) 

        for i in range(I):
                c = estimate_colors(FuncM, FuncN, img, L[i])
                set_color(img, L[i], c)
                print c

# Read image frames from camera source and diplsat segmented frames
def main():
        try: video_src = sys.argv[1]
        except: video_src = 0

        # 2 segments by default
        try: n = int(sys.argv[2])
        except: n = 3

        camera = cv2.VideoCapture(video_src)
        FuncR = get_FuncR()
        FuncM = get_FuncM()
        FuncN = get_FuncN()

        while True:
                # grab the current frame and initialize
                (grabbed, img) = camera.read()

                # if the frame could not be grabbed, then we have reached the end
                # of the video
                if not grabbed:
                        break
                cv2.imshow("input", img)

                btquant(FuncR, FuncM, FuncN, img, n)

                cv2.imshow("cluster", img)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                        break
        camera.release()
        cv2.destroyAllWindows()

main()
