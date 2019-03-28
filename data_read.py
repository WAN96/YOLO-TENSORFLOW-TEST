import  cv2
import  numpy as np
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import  os
import  sys
import  pickle
import  random
from config import  cfg
import tensorflow as tf

class kitti_voc(object):
    def __init__(self):
        self.image_width = cfg.IMAGESIZE
        self.image_height = cfg.IMAGESIZE
        self.batchsize = cfg.BATCHSIZE
        self.data_path ='/media/wx/File/KITTIVOC/'
        self.classes = cfg.CLASSES
        self.fliped = False
        self.cellsize=cfg.CELLSIZE
        self.phase = 'train'
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.cache_path= cfg.CACHEPATH
        self.rebuild = False
        self.id = 0
        self.epoch = 0
        self.index=list( np.arange(0, 7481,step=1))
        #self.gt_labels = None

    # get data from the kitti dataset which is in voc version
    def get_data(self):
        image = np.zeros((self.batchsize,self.image_height,self.image_width,3))
        labels = np.zeros((self.batchsize,self.cellsize,self.cellsize,8))
        count = 0
        gt_labels = self.data_prepare()
        #print(len(gt_labels))
        #print(self.index)
        while count< self.batchsize:
            imname  = gt_labels[self.index[self.id]]['imname']
            flipped = gt_labels[self.index[self.id]]['flipped']
            labels[count,:,:] = gt_labels[self.index[self.id]]['label']
            image[count,:,:,:] = self.read_img(imname,flipped)
            count = count +1
            self.id = self.id +1
            if self.id >=self.batchsize:
               self.id = 0
               print('epoch:{}'.format(self.epoch))
               self.epoch = self.epoch+1
               random.shuffle(self.index) #when get one epoch random the index
               #print(self.index)
        return  image, labels

    # read image
    def read_img(self, imname, flipped):
        image =cv2.imread( imname)
        image = cv2.resize(image,(self.image_width,self.image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image/255.0)*2.0 -1.0
        if flipped:
            image = image[:,::-1,:]
        return  image

    # return the ground truth lable 
    def  data_prepare(self):
         index = []
         cache_file = os.path.join(self.cache_path, 'kitti_' + self.phase + '_gt_labels.pkl')
         with open(self.data_path+"ImageSets/Main/"+self.phase+'.txt')as f:
            for line in f:
                index.append(line.strip('\n'))

         if os.path.isfile(cache_file) and not self.rebuild:
             #print('Loading gt_labels from: ' + cache_file)
             with open(cache_file, 'rb') as f:
                 gt_labels = pickle.load(f)
             return gt_labels
         print('Processing gt_labels from: ' + self.data_path)

         if not os.path.exists(self.cache_path):
             os.makedirs(self.cache_path)
         gt_labels=[]
         for i in index:
             label, obnum , imgname = self.load_kitti_annotation( i )
             if  obnum == 0:
                 continue
             gt_labels.append ({'imname':imgname,
                                         'label': label,
                                         'flipped': False})
         print('Saving gt_labels to: ' + cache_file)
         with open(cache_file, 'wb') as f:
             pickle.dump(gt_labels, f)
         return gt_labels



    def  load_kitti_annotation(self, index):
        imname = os.path.join(self.data_path, 'JPEGImages', index+'.jpg')
        image = cv2.imread(imname)
        h_ratio = self.image_height / image.shape[0]  #ratio of an image
        w_ratio = self.image_width / image.shape[1]
        #cv2.imshow(' ',image)
        #cv2.waitKey(0)
        label = np.zeros((self.cellsize,self.cellsize,8))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        DOMTree = ET.parse(filename)
        objs = DOMTree.findall('object')
        for ob in objs:
            bbox = ob.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float( bbox.find('xmin').text )- 1) * w_ratio, self.image_width - 1), 0)
            y1 = max(min((float( bbox.find('ymin').text )- 1) * h_ratio, self.image_height - 1), 0)
            x2 = max(min((float( bbox.find('xmax').text )- 1) * w_ratio, self.image_width - 1), 0)
            y2 = max(min((float( bbox.find('ymax').text )- 1) * h_ratio, self.image_height - 1), 0)
            if ob.find('name').text.lower().strip()== 'dontcare':
                continue
            cls_ind = self.class_to_ind[ob.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            #print(boxes)
            x_ind = int(boxes[0] * self.cellsize / self.image_width) #find the postion in imagecell
            #print(x_ind)
            y_ind = int(boxes[1] * self.cellsize / self.image_height)
            #print(y_ind)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs), imname


# return label:[batchsize, cellsize, cellsize, 8]  image:[ batchsize, imagesize, imagesize, 3]
'''
ima = cv2.imread("/media/wx/File/KITTIVOC/JPEGImages/002000.jpg")
#im=cv2.resize(ima,(500,500))
data = kitti_voc()
w_ratio = ima.shape[1]/500
h_raito = ima.shape[0]/500
y,d,z= data.load_kitti_annotation('002000')
print(y.shape)
for i in range(y.shape[0]):
    for j in  range(y.shape[1]):
        x=y[i,j,1]
        iy = y[i,j,2]
        w = y[i, j,3]
        h = y[i,j,4]
        xmin = (x- w/2)*w_ratio
        ymin = (iy - h/2)*h_raito
        xmax = (x +w/2)*w_ratio
        ymax = (iy+h/2)*h_raito
        cv2.rectangle(ima, (int(xmin),int(ymin)), (int(xmax),int(ymax)),color=(0,0,255))

#im2=cv2.resize(im,(1278,372))
cv2.imshow(' ',ima)
#cv2.waitKey(0)

data = kitti_voc()
x=data.get_data()
'''

#print(x)

#debug test

