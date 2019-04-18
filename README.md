# YOLO-TENSORFLOW-TEST

This is a simple test on YOLO, I build the net work by referencing https://github.com/CharlesShang/TFFRCNN and https://github.com/hizhangp/yolo_tensorflow

---

#tensorflow 1.2

#python 3.5

#opencv

#KITTI databse

---

I use KITTI dataset to train this network
Before training, I change the dataset format into voc vesrion by using the code from  https://github.com/CharlesShang/TFFRCNN

If you want to train the network, just run

  `python train_net.py`
  
you can see the training process by using tensorboard

  `tensorboard --logdir log`
  
  
![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/blob/master/img/Screenshot%20from%202019-03-25%2019-25-53.jpg)


![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/blob/master/img/Screenshot%20from%202019-03-28%2016-56-20.jpg)


![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/blob/master/img/Screenshot%20from%202019-03-28%2016-56-32.jpg)
