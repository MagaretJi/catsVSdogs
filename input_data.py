import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定要使用的显卡的编号0-3
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True # 允许显存自适应增长
tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0 # 最多使用30%显存，看情况设置
sess = tf.Session(config=tf_config)

def get_files(file_dir,is_random=True):
    image_list=[]
    label_list=[]
    dog_count=0
    cat_count=0
    for file in os.listdir(file_dir):
        name=file.split(sep='.')
        if(name[0]=='cat'):
            image_list.append(file_dir+file)
            label_list.append(0)
            cat_count+=1
        else:
            image_list.append(file_dir+file)
            label_list.append(1)
            dog_count+=1
    print('%d cats and %d dogs'%(cat_count,dog_count))

    image_list=np.asarray(image_list)
    label_list=np.asarray(label_list)

    if is_random:
        rnd_index=np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list=image_list[rnd_index]
        label_list=label_list[rnd_index]

    return image_list,label_list

def get_batch(train_list,image_size,batch_size,capacity,is_random=True):

    #生成队列
    input_queue=tf.train.slice_input_producer(train_list,shuffle=False)
    image_train=tf.read_file(input_queue[0])
    image_train=tf.image.decode_jpeg(image_train,channels=3)
    image_train=tf.image.resize_images(image_train,image_size)
    image_train=tf.cast(image_train,tf.float32)/255

    #图片标签
    label_train=input_queue[1]

    if is_random:
        image_train_batch,label_train_batch=tf.train.shuffle_batch([image_train,label_train],batch_size=batch_size,capacity=capacity,
                                                                   min_after_dequeue=100,num_threads=2)
    else:
        image_train_batch,label_train_batch=tf.train.shuffle_batch([image_train,label_train],batch_size=1,capacity=capacity,min_after_dequeue=100,
                                                                   num_threads=1)

    return image_train_batch,label_train_batch



if __name__=='__main__':
    train_dir='/home/margaret/dogVScat/data/train/'
    train_list=get_files(train_dir,is_random=True)
    image_train_batch, label_train_batch=get_batch(train_list,[256,256],1,200,False)

    with tf.Session() as sess:
        i=0
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        try:
            for step in range(10):
                if coord.should_stop():
                    break
                image_batch,label_batch=sess.run([image_train_batch,label_train_batch])
                if label_batch[0]==0:
                    label='cat'
                else:
                    label='dog'
                    plt.imshow(image_batch[0])
                    plt.show()

        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(threads=threads)


