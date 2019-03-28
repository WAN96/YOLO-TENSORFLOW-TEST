import  tensorflow as tf
from cal_iou import  calc_iou
import  numpy as np
import  cv2

def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

#copy from TFFRCNN-master this is a layer decorator,which makes the net easy to read 

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #print(name)
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable= True):
        self.input = []
        self.layers = dict(inputs)
        self.trainable = trainable


    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = [ ]
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    tf.summary.image('input', layer)
                    #print (layer)
                except KeyError:
                    print (self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    ########  convolution operation  #######
    @layer
    def conv_op(self,input_op ,kh, kw, n_out, dh, dw,name):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([kh, kw,n_in, n_out], stddev=0.1),name=scope+'weights')
            conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw, 1), padding='SAME')
            bias_inital = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_inital,trainable=True, name=scope+'biases')
            z= tf.nn.bias_add(conv,biases)
            output = leak_relu(z)
        return output

    ######## fully coneccted layer  #########
    @layer
    def  fc_op(self,input_op, n_out, name,activation = None ):
        #n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            if isinstance(input_op, tuple):
                input_op = input_op[0]

            input_shape = input_op.get_shape()
            #print(input_shape[1:])
            if input_shape.ndims == 4:
                n_in = 1
                for d in input_shape[1:].as_list():
                    n_in *= d
                input_op = tf.reshape(tf.transpose(input_op,[0,3,1,2]), [-1, n_in])
            else:
                input_op, n_in = (input_op, int(input_shape[-1]))

            kernel = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1),name=scope+'weights')
            #print(kernel.shape)
            bias_inital = tf.constant(0.0, shape=[n_out, ], dtype=tf.float32)
            biases = tf.Variable(bias_inital,trainable=True, name=scope+'biases')
            input_op = tf.reshape(input_op, (-1, n_in))
            output = tf.matmul(input_op, kernel)+ biases
            if activation:
                output =activation(output)
            #p+=[kernel, biases]
        return output

    ###### pooling layer  ########
    @layer
    def  maxpool_layer(self,x,  pool_size, stride, name):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],strides=[1, stride, stride, 1], padding="SAME", name=name)
        return output


    def loss_layer(self,scope='loss_layer'):

        with tf.variable_scope(scope):
            predict_class = tf.reshape (self.out [:, :self.classboudry], [self.batch_size,self.cell_size, self.cell_size, 3])
            #print(tf.shape(predict_class))
            predict_conf = tf.reshape(self.out  [:, self.classboudry:self.scoreboundry], [self.batch_size, self.cell_size, self.cell_size,self.boxes_per_cell])
            predict_boxes = tf.reshape(self.out  [:, self.scoreboundry:], [ self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell,4])

            conf=tf.reshape ( self.label[..., 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes =tf.reshape( self.label[..., 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes= tf.tile(boxes, [1, 1, 1,self.boxes_per_cell, 1])/self.image_size
            classes = self.label[..., 5:]

            predict_boxtran = tf.stack([(predict_boxes[..., 0] ) / self.cell_size, (predict_boxes[..., 1] ) / self.cell_size, tf.square(predict_boxes[..., 2]), tf.square(predict_boxes[..., 3])], axis=-1)
            boxes_tran = tf.stack([boxes[..., 0], boxes[..., 1], tf.square(boxes[..., 2]), tf.square(boxes[..., 3])], axis= -1)

            #print(predict_boxes.shape)
            #print('boxes:',boxes)


            iou_predict_truth = calc_iou(predict_boxtran, boxes)

            object_mask =  tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * conf

            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask


            # class_loss
            class_delta = conf * (predict_class - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale
            #print(class_loss.shape)
            # object_loss
            object_delta = object_mask * (predict_conf - iou_predict_truth)
            object_loss = tf.reduce_mean( tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_conf
            noobject_loss = tf.reduce_mean( tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean( tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale


        return  class_loss, object_loss, noobject_loss, coord_loss

'''
def  build_imagesummary(self, scope='image'):
        with tf.name_scope(scope) as scope:
            predict_boxes = tf.reshape(self.out[:, self.scoreboundry:],
                                       [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            for k in  range(self.batch_size):
                for i in range(self.layers['img'].shape[0]):
                    for j in range(self.layers['img'].shape[1]):
                         ima = np.reshape(self.image[k], (self.imagesize,self.imagesize, 3))
                         x = self.image[k,i, j, 1]
                         iy = self.image[k,i, j, 1]
                         w =  self.image[k,i, j, 1]
                         h =  self.image[k,i, j, 1]
                         xmin = (x - w / 2)
                         ymin = (iy - h / 2)
                         xmax = (x + w / 2)
                         ymax = (iy + h / 2)
                         cv2.rectangle(ima, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255))
                         tf.summary.image('out', ima)
'''
