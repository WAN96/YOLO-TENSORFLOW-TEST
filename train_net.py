import  numpy as np
import os
import  tensorflow as tf
from data_read import  kitti_voc
from yolonet import  YoloNet
import  time
from config import  cfg

class Solver(object):
    def __init__(self, sess,  network,outpudir, logdir):
        self.net = network()
        self.outputdir = outpudir
        self.logdir = logdir
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(logdir=logdir,graph=sess.graph,flush_secs=5)
        self.batch_size = 3

    def train(self, sess, iters):
        # add loss
        class_loss, object_loss, noobject_loss, coord_loss = self.net.loss_layer()
        #self.net.build_imagesummary()

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)

        loss = tf.losses.get_total_loss()

        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)
        summary_op = tf.summary.merge_all()

        # optimizer
        opt = tf.train.AdamOptimizer(cfg.LEARNINGRATE)
        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            gradient = tf.gradients(loss, tvars)
            grads, norm = tf.clip_by_global_norm(gradient , 10.0)
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        #restore_iter = 0
        get = kitti_voc()

        for iter in range (iters):
            print('iteration:',iter)
            timestart = time.time()
            img, data = get.get_data()
            #print('Datashape;',data.shape)
            
            feed_dict = {self.net.img: img,
                         self.net.gt_lable: data}
            fetch_list = [class_loss, object_loss,noobject_loss, coord_loss , loss, summary_op,train_op]

            class_loss_result, object_loss_result, noobject_loss_result, coord_loss_result, loss_result, summary_str,_ = sess.run(fetches= fetch_list, feed_dict=feed_dict)
            
            timecost = time.time() - timestart

            #if  iter % self.batch_size == 0 :
                #print('epoch:{}/')

            print('Class_loss:{}, object_loss:{}, noobject_loss:{}, coord_loss:{}, total_loss:{}'.format(class_loss_result,object_loss_result, noobject_loss_result, coord_loss_result , loss_result))
            print('One iter cost time:{}'.format(timecost))

            if iter % 1 ==0 :
                self.saver.save(sess, 'model/yolo_net', global_step=iter)
                print('saving model to dir:{}'.format('model'))
            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())


def train_net(network ,  output_dir, log_dir,  max_iters=4000):
    with tf.Session() as sess:
        sw = Solver(sess, network,output_dir, logdir=log_dir)
        print('Solving...')
        sw.train(sess, max_iters)
        print ('done solving')


# train the yolo
logdir ='/home/wx/YOLO-TENSORFLOW/log'
outputdir = '/home/wx/LEARN'
train_net(YoloNet, outputdir, logdir)
