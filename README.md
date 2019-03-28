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
  
  
![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/tree/master/img/tensorboard1.png)


![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/tree/master/img/tensorboard2.png)


![Image text](https://github.com/WAN96/YOLO-TENSORFLOW-TEST/tree/master/img/tensorboard3.png)
