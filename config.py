from easydict import  EasyDict as  edict
import os
import os.path as osp

_C = edict()
cfg = _C
# FOR TRAIN
_C.BATCHSIZE = 3
_C.NOOBJECTSCALE = 1.0
_C.OBJECTSCLAE  = 1.0
_C.CLASSSCALE = 2.0
_C.COORDSCALE = 5.0
_C.LEARNINGRATE = 0.0001

_C.CLASSES = ['car', 'pedestrian', 'cyclist']

# FOR NET
_C.IMAGESIZE = 500
_C.NUMCLASS = 3
_C.CELLSIZE = 10
_C.BOXPERCELL = 3


_C.CACHEPATH = osp.join(os.getcwd(), 'cache/')