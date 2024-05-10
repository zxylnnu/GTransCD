import numpy as np
import scipy.io as sio
from sklearn import preprocessing

class SegmentMap(object):
    def __init__(self):
        segs = sio.loadmat(r'XXX.mat')
        self.segs = segs['segmentmaps']

    def getHierarchy(self):
        segs=self.segs
        layers, h, w = self.segs.shape
        segs=np.concatenate([np.reshape( [i for i in range(h*w)],[1,h,w] ), segs],axis=0)
        layers=layers+1
        
        Q_mat=[]
        
        for i in range(layers-1):
            Q=np.zeros([np.max(segs[i])+1,np.max(segs[i+1])+1])
            l1=np.reshape(segs[i],[-1])
            l2=np.reshape(segs[i+1],[-1])
            for x in range(h*w):
                if Q[ l1[x] ,l2[x]]!=1:
                    Q[ l1[x] ,l2[x]]=1
            Q_mat = Q
        
        A_mat = []

        superpixelLabels=self.segs

        for l in range(len(superpixelLabels)):
            segments = np.reshape(superpixelLabels[l], [h, w])
            superpixel_count = int(np.max(superpixelLabels[l])) + 1
            A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)

            for i in range(h - 1):
                for j in range(w - 1):
                    sub = segments[i:i + 2, j:j + 2]
                    sub_max = np.max(sub).astype(np.int32)
                    sub_min = np.min(sub).astype(np.int32)

                    if sub_max != sub_min:
                        idx1 = sub_max
                        idx2 = sub_min
                        if A[idx1, idx2] != 0: continue
                        pix1 = Q[idx1]
                        pix2 = Q[idx2]
                        diss = np.exp(-np.sum(np.square(
                            pix1 - pix2)) / 10 ** 2)
                        A[idx1, idx2] = A[idx2, idx1] = diss
                    A = preprocessing.scale(A, axis=1)
                    A = A + np.eye(superpixel_count)
            A_mat = A
        return Q_mat, A_mat
