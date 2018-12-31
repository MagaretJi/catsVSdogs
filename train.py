import time
from input_data import *
from model import *
import tensorflow as tf
import numpy as np



def train():
    file_dir = '/home/margaret/dogVScat/data/train/'
    logs_dir = '/home/margaret/dogVScat/train_logs_1'
    N_ClASS=2
    IMG_SIZE=208
    BATCH_SIZE=8
    CAPACITY=200
    MAX_STEP=10000
    LEARNING_RATE=1e-4

    sess=tf.Session()

    train_list=get_files(file_dir,is_random=True)
    image_train_batch,label_train_batch=get_batch(train_list,[IMG_SIZE,IMG_SIZE],BATCH_SIZE,CAPACITY,True)
    train_logits=net(image_train_batch)
    train_loss=losses(train_logits,label_train_batch)
    train_acc=evaluation(train_logits,label_train_batch)

    train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    var_list=tf.trainable_variables()
    parse_count=tf.reduce_sum([tf.reduce_prod(v.shape)for v in var_list])
    print('参数数目：%d'%sess.run(parse_count),end='\n\n')

    saver=tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    s_t=time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            _,loss,acc=sess.run([train_op,train_loss,train_acc])

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', acc)

            if step%100==0:
                runtime=time.time()-s_t
                print('Step:%6d,loss:%.8f,accuracy:%.2f,time:%.2fs,time left:%.2fhours'%(step,loss,acc*100,runtime,(MAX_STEP-step)*runtime/360000))
                s_t=time.time()

            if step%1000==0 or step==MAX_STEP-1:
                checkpoint_path=os.path.join(logs_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

def get_one_image():
    IMG_SIZE=208
    BATCH_SIZE=1
    CAPACITY=200
    MAX_STEP=10

    test_dir='/home/margaret/dogVScat/data/test1/1.jpg'
    logs_dir='/home/margaret/dogVScat/logs_1'

    sess=tf.Session()
    train_list=get_files(test_dir,is_random=True)
    image_train_batch,label_train_batch=get_batch(train_list,[IMG_SIZE,IMG_SIZE],BATCH_SIZE,CAPACITY,True)
    train_logits=net(image_train_batch)
    train_logits=tf.nn.softmax(train_logits)

    sess.run(tf.global_variables_initializer())

    #载入检查点
    saver=tf.train.Saver()
    print('\n载入检查点')
    ckpt=tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('载入成功，global_step=%s\n'%global_step)
    else:
        print('没有找到检查点')
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            image,prediction=sess.run([image_train_batch,train_logits])
            max_index=np.argmax(prediction)
            if max_index==0:
                label='%.2f%% is a cat'%(prediction[0][0]*100)
            else:
                label='%.2f%% is a dog'%(prediction[0][1]*100)

    except tf.errors.OutOfRangeError:
        print('Done')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__=='__main__':
    train()
    #eval()
