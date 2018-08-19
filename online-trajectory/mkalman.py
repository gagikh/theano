import cv2, numpy as np
import time

meas=[]
pred=[]
N = 800
M = 800

record = False
trajectory_id = 0

frame = np.zeros((N, M,3), np.uint8) # drawing canvas
mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction


def onmouse(k,x,y,s,p):
    global mp,meas
    mp = np.array([[np.float32(x)],[np.float32(y)]])
    meas.append((x,y))

def paint():
    global frame,meas,pred
    for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(0,100,0))
    for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))

def reset():
    global meas,pred,frame
    meas=[]
    pred=[]
    frame = np.zeros((N, M,3), np.uint8)
    print "Stopping record"
    record = False

cv2.namedWindow("kalman")
cv2.setMouseCallback("kalman",onmouse);

#kalman = cv2.KalmanFilter(4,2)
#kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
#kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
#kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03


kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([
				[1,0,1,0],
				[0,1,0,1],
				[0,0,1,0],
				[0,0,0,1]
			    ],np.float32)
kalman.processNoiseCov = np.array([
				[1,0,0,0],
				[0,1,0,0],
				[0,0,1,0],
				[0,0,0,1]
			    ],np.float32) * 0.03

start = time.time();
while True:
    kalman.correct(mp)
    tp = kalman.predict()
    pred.append((int(tp[0]),int(tp[1])))
    paint()
    if record:
	with open('mrecord.txt', 'a') as the_file:
	    x = str(tp[0][0])
	    y = str(tp[1][0])
	    t = str(time.time() - start)
	    s = x + "," + y + "," + t + "," + str(trajectory_id) + "\n";
	    the_file.write(s)
    cv2.imshow("kalman",frame)
    k = cv2.waitKey(30) &0xFF
    if k == 27: break
    if k == 32: reset()
    if k == ord('r'):
	print "Starting record"
	record = True
	trajectory_id = trajectory_id + 1
