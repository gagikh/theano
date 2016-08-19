import theano, csv
import numpy as np

from theano import tensor as T
from random import randint, shuffle

def load_data(filename):
	M = 4
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		# As the data-labels are arranged in sequential way
		# we make it randomly distributed via in-place shuffle
		shuffle(dataset)
		N = len(dataset)

		train = np.zeros((N, M))
		labels = np.zeros(N, dtype=np.int32)
		labels_map = dict()
		idx = 0

		for x in range(N):
			for y in range(M):
				train[x][y] = float(dataset[x][y])

			l = dataset[x][M]
			if l in labels_map:
				labels[x] = labels_map[l]
			else:
				labels[x] = idx
				labels_map[l] = idx
				idx = idx + 1

		return (train, labels, labels_map)

def edistance(x1, x2):
	d = x1 - x2
	z = T.sqrt(T.dot(d, d))
	return z

# returns the labels of K neighbors
def kneighbors(dist):
	train  = T.matrix("train")
	labels = T.ivector("labels")
	test_sample = T.vector("sample")
	k = T.iscalar("k")
	l = T.ivector("l")

	r, u = theano.scan(lambda row: dist(row, test_sample), sequences=train)
	z = T.argsort(r)
	i = z[0:k]
	l = labels[i]
	return theano.function(inputs=[train, labels, test_sample, k], outputs=l)

# Returns a label by voting nearest point labels
def get_nearest():
	l = T.ivector("l")
	z = T.extra_ops.bincount(l)
	a = T.argmax(z)
	o = l[a]
	return theano.function(inputs=[l], outputs=o)

def main():
	k = 5
	train, labels, labels_map = load_data('iris.data')

	# Create theano functions
	nfcn = kneighbors(edistance)
	gfcn = get_nearest()

	n = len(labels)
	n = randint(n / 2, n)

	print "split len = ", n
	print "labels map len = ", len(labels_map)
	print "data-set len = ", len(train)
	print "k = ", k

	# Split train data
	t1 = train[0:n-1]
	t2 = train[n:]
	l1 = labels[0:n-1]
	l2 = labels[n:]
	correct = 0

	for idx in range(len(t2)):
		l = nfcn(t1, l1, t2[idx], k)
		m = gfcn(l)
		g = l2[idx]
		# estimate the results
		if m == g:
			correct = correct + 1

	p = (correct / float(len(t2))) * 100.0
	print "test results:", p, "%"

main()
