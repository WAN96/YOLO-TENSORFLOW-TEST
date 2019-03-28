import  tensorflow as tf
from utils import  Network
from cal_iou import calc_iou
from config import cfg

class YoloNet(Network):
    def __init__(self, trainable = True):
        self.inputs =[ ]
        self.num_class = cfg.NUMCLASS
        self.image_size = cfg.IMAGESIZE
        self.cell_size = cfg.CELLSIZE
        self.boxes_per_cell = cfg.BOXPERCELL
        self.batch_size = cfg.BATCHSIZE
        self.img = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.num_class], name='img')
        self.gt_lable = tf.placeholder(tf.float32, [None, self.cell_size,self.cell_size, 8], name= 'gt_label')
        self.layers = dict({'img': self.img, "gt_label": self.gt_lable})
        self.noobject_scale = cfg.NOOBJECTSCALE
        self.class_scale = cfg.CLASSSCALE
        self.coord_scale = cfg.COORDSCALE
        self.object_scale = cfg.OBJECTSCLAE
        self.trainable = trainable
        self.classboudry = self.cell_size*self.cell_size*3
        self.scoreboundry =  self.classboudry+ self.cell_size*self.cell_size*(self.boxes_per_cell)
        self.label = self.gt_lable
        self.setup()
        self.out  = self.get_output('fc3')

    def setup(self):

        ( self.feed('img')
         .conv_op(10, 10, 64, 1, 1,name='conv1')
         .maxpool_layer(2,2)
         .conv_op( 3,3, 192, 1, 1,name='conv2')
         .maxpool_layer(2,2)
         .conv_op(1, 1, 128, 1, 1, name='conv3')
         .conv_op(3, 3, 256, 1, 1, name='conv4')
         .conv_op(1, 1, 256, 1, 1, name='conv5')
         .conv_op(3, 3, 512, 1, 1, name='conv6')
         .maxpool_layer(2,2)
         .conv_op(1, 1, 256, 1, 1, name='conv7')
         .conv_op(3, 3, 512, 1, 1, name='conv8')
         .conv_op(1, 1, 256, 1, 1, name='conv9')
         .conv_op(3, 3, 512, 1, 1, name='conv10')
         .conv_op(1, 1, 256, 1, 1, name='conv11')
         .conv_op(3, 3, 512, 1, 1, name='conv12')
         .conv_op(1, 1, 256, 1, 1, name='conv13')
         .conv_op(3, 3, 512, 1, 1, name='conv14')
         .conv_op(1, 1, 512, 1, 1, name='conv15')
         .conv_op(3, 3, 1024, 1, 1, name='conv16')
         .maxpool_layer(2,2)
         .conv_op(1, 1, 512, 1, 1, name='conv17')
         .conv_op(3, 3, 1024, 1, 1, name='conv18')
         .conv_op(1, 1, 512, 1, 1, name='conv19')
         .conv_op(3, 3, 1024, 1, 1, name='conv20')
         .conv_op(3, 3, 1024, 1, 1, name='conv21')
         .conv_op(3, 3, 1024, 2, 2, name='conv22')
         .conv_op(3, 3, 1024, 1, 1, name='conv23')
         .conv_op(3, 3, 1024, 1, 1, name='conv24')
         .fc_op(512 ,name='fc1')
         .fc_op(4096, name='fc2')
         .fc_op(10*10*18, name='fc3')
          )

