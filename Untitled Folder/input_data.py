
# coding: utf-8

# In[1]:



#这个是没有生成任何的文件或者结果，只是导入了两个函数，生成的image_batch和label_batch给下部分使用
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

#######################生成图片的路径和标签的list
train_dir = r'F:\anacondapython\Untitled Folder\gen_pic'

horse = []
label_horse = []

bus = []
label_bus = []

long = []
label_long = []

flower = []
label_flower = []

elephant = []
label_elephant = []


########################第一步：获取路径下的所有图片的路径名，存放到对应的列表当中，同事贴上标签，存放到label列表当中
def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'\\horse'):
        horse.append(file_dir+'/horse'+'/'+file)
        label_horse.append(0)

    for file in os.listdir(file_dir+'\\bus'):
        bus.append(file_dir+'/bus'+'/'+file)
        label_bus.append(1)

    for file in os.listdir(file_dir+'\\long'):
        long.append(file_dir+'/long'+'/'+file)
        label_long.append(2)

    for file in os.listdir(file_dir+'\\flower'):
        flower.append(file_dir+'/flower'+'/'+file)
        label_flower.append(3)

    for file in os.listdir(file_dir+'\\elephant'):
        elephant.append(file_dir+'/elephant'+'/'+file)
        label_elephant.append(4)

    ############第二步:对生成的图片路径和标签list做打乱处理，把样本组合起来形成一个list(image和label)
    image_list = np.hstack((horse,bus,long,flower,elephant))
    label_list = np.hstack((label_horse,label_bus,label_long,label_flower,label_elephant))
    ###利用shuffle打乱顺序
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    ###将所有的img和label转换成list
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])
    ###将所得的list分为两部分，一部分用来训练tra，一部分用来测试val
    ###ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) #测试样本数
    n_train = n_sample-n_val   #训练样本数


    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]


    return tra_images,tra_labels,val_images,val_labels
##########运行到这里是返回 4个list 训练集的图片、标签    交叉验证集的图片、标签   这里返回的是存放图片的地址，不是实际的图片，需要下面的gei_batch来读取图片


#############################生成batch
#第一步将上面生成的list传入get_batch(),转换类型，产生一个输入队列queue，因为img和label是分开的，
#所以使用tf.train.slice_input_producer(),然后使用tf.read.file()从队列中读取图像
#image_w，image_h 设置好固定的图像高度和宽度
#设置batch_size,每个batch要放多少张图片
#capacity：一个队列最大多少
def get_batch(image,label,image_w,image_h,batch_size,capacity):
    #转换类型
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    #制造一个输入队列
    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  #从队列中读取图片
    #第二步:将图像解码，不同类型的图像不能混在一起，要么只用jpg，要么只用png等
    image = tf.image.decode_jpeg(image_contents,channels=3)#因为使用的是jpg格式，所以就用jpeg的解码器
    #第三步：数据预处理，对图像进行旋转，缩放，裁剪，归一化等操作。
    image = tf.image.resize_image_with_crop_or_pad(image,image_w,image_h)
    image = tf.image.per_image_standardization(image)
    #第四步：就是生成batch
    #image_batch: 4D tensor [batch_size,width,height,3]  dtype=tf.float32
    #label_batch: 1D tensor [batch_size] dtype=tf.int32
    image_batch,label_batch = tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads = 32,        
                                             capacity = capacity)
    ######################################################## num_threads = 32是代表线程,可以改成64
    #重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    return image_batch,label_batch


    #########运行到这里就是获得两个batch，这两个batch就是传入神经网络的数据
    ###怎么传进去呢？就是 xs,ys = get_batch()


