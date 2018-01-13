__author__ = 'JD07'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import collections
import functools
import threading
import argparse
import datetime
import logging
import socket
import uuid
import json
import time
import sys
import cv2
import os
import re

from local_utils import  data_utils
from icdar import restore_rectangle
from model import east_model #east网络参数
from model import crnn_model #crnn网络参数

import lanms

FLAGS = None#命令行参数全局变量


def getfilelist(path):
    '''
        该函数用于寻找指定路径下所有图像，并返回路径list
        输入：
            path：路径
        返回：
            filelist：图片路径组成的list
    '''
    filelist = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)):#确认是否是文件
            if i.endswith('.jpg') or i.endswith('.png') or  i.endswith('.jpeg'):#确认是否是jpg或png
                filelist.append(os.path.join(path, i))
        else:
            filelist+=(getfilelist(os.path.join(path, i)))
    return filelist

def draw_illu(illu, rst):
    '''
        在输入图像illu上，按rst指定的坐标画框
        输入：
            illu：图像
            rst：包含了RBOX坐标的有序dict
        返回：
            illu：画好框的图像
    '''
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)#按顺序两个元素组成一列，形成四个点的格式
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def cropImg(dirPath, img, rst):
    '''
        该函数用于将图像按照RBOX坐标进行切割，并将切割后的图像保存到指定路径下
        输入：
            dirPath：指定保存的路径
            img：切割原图
            rst：包含了RBOX坐标的有序dict
        返回：
    '''
    for i,t in enumerate(rst['text_lines']):
        x0 = int(min(t['x0'], t['x1'], t['x2'], t['x3']))
        x1 = int(max(t['x0'], t['x1'], t['x2'], t['x3']))
        y0 = int(min(t['y0'], t['y1'], t['y2'], t['y3']))
        y1 = int(max(t['y0'], t['y1'], t['y2'], t['y3']))
        offset = (x1-x0)//8 #由于定位存在问题，导致RBOX框的水平宽度经常不足
        rstImg = img[y0:y1, max(0, x0-offset):x1+offset]
        rstName = str(i) + '.jpg'
        rstPath = os.path.join(dirPath, rstName)
        cv2.imwrite(rstPath, rstImg)


def save_result(i, img, rst):
    '''
        对输入的图像执行切割，并将其rst信息与切割后的图片保存到独立的文件夹下
        输入：
            i：用于产生独立文件夹
            img：图像
            rst：包含了RBOX坐标等信息的有序dict
        返回：
            rst：
            dirPath：保存切割结果的独立文件夹路径
    '''
    dirPath = os.path.join(FLAGS.cropPath, str(i))
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    #save crop
    cropImg(dirPath, img.copy(), rst)

    # save json data
    output_path = os.path.join(dirPath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    return rst, dirPath

def restore_rectangle_rbox(origin, geometry):
    '''
        恢复RBOX框，非我编写，需要花时间理解
    '''
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def resize_image(im, max_side_len=2400):
    '''
        将图像resize，满足指定要求
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
        通过score_mao和geo_map来恢复text_boxes
        输入：
            score_map:
            geo_map:
            timer:用来记录各步骤耗时的有序字典
            score_map_thresh:threshhold for score map
            box_thresh:threshhold for boxes
            nms_thres:threshold for nms
        输出：
            boxes：
            timer：
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle_rbox(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def getPredictor():
    #该函数用于从ckpt文件中恢复网络，并返回一个可以得到相应图像结果的predictor函数
    with tf.Graph().as_default() as net1_graph:
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = east_model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver1 = tf.train.Saver(variable_averages.variables_to_restore())
    
    sess1 = tf.Session(graph=net1_graph)       
    saver1.restore(sess=sess1, save_path=FLAGS.eastWeightsPath)

    def predictor(img):
        '''
        用于执行网络前向传播的函数
        输入：
            img：输入网络的图像
        返回：
            {
            'text_lines': 
                [
                    {
                        'score': ,
                        'x0': ,
                        'y0': ,
                        'x1': ,
                        ...
                        'y3': ,
                    }
                ],
            'rtparams': 
                {  
                    'image_size': ,
                    'working_size': ,
                },
            'timing': 
                {
                    'net': ,
                    'restore': ,
                    'nms': ,
                }
            }

        '''
        start_time = time.time()
        #产生有序字典rtparams和timer，用于记录各项参数以及各部所花时间
        rtparams = collections.OrderedDict() 
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])
        
        #resize图像并记录处理后的图像尺寸
        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        
        #执行前向传播
        start = time.time()        
        score, geometry = sess1.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start
        #计算boxes
        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
            }
        return ret

    return predictor

def getRecognize():
    #该函数用于从ckpt文件中恢复网络，并返回一个可以识别图像内容的recognize函数
    with tf.Graph().as_default() as net2_graph:
        inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

        net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=20)

        with tf.variable_scope('shadow'):
            net_out = net.build_shadownet(inputdata=inputdata)

        decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)
        decoder = data_utils.TextFeatureIO()
        
        saver2 = tf.train.Saver()

    sess2 = tf.Session(graph=net2_graph)   
    saver2.restore(sess=sess2, save_path=FLAGS.crnnWeightsPath)

    def recognize(path):
        '''
            对指定路径下的所有图像执行crnn网络识别，并在命令行上显示结果
            输入：
                path：指定数据集路劲
            返回：
        '''

        resultPath = os.path.join(path, 'result.txt')
        imageList = getfilelist(path)
        f=open(resultPath, 'w')

        for i in range(len(imageList)):
            image = cv2.imread(imageList[i], cv2.IMREAD_COLOR)
            #image = cv2.imread(imageList[i], 0)
            image = cv2.resize(image, (100, 32))
            #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,23,10)
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.expand_dims(image, axis=0).astype(np.float32)
            preds = sess2.run(decodes, feed_dict={inputdata: image})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])
            f.write('{:s} {:s}'.format(os.path.split(imageList[i])[1], preds[0]))
            f.write('\n')
        f.close()
        '''
        for imagePath in imageList:
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 32))
            image = np.expand_dims(image, axis=0).astype(np.float32)
            preds = sess2.run(decodes, feed_dict={inputdata: image})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])
            print('Predict image {:s} label {:s}'.format(os.path.split(imagePath)[1], preds[0]))
        '''


    return recognize

def tcplink(sock, addr, predictor, recognize):
    '''
        socket监听线程
        输入：
            sock：
            addr：
            predictor：east前向传播函数
            recognize：crnn前向传播函数
    '''
    print('Accept new connection from %s:%s...' % addr)
    imagePath = str()

    #循环接受client数据，直到收到$为止
    while True:
        data = sock.recv(1024)
        data = data.decode()
        imagePath = imagePath + data#使用字符串合并，防止数据超过1024字节
        if data.find('$') or not data:
            break
    #对字符串进行截取，舍弃$后的部分
    imagePath = imagePath.split('$')[0]
    #对字符串进行截取，获得文件名
    i = os.path.basename(imagePath).split('.')[-2]

    #确认目标图像存在
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
    
    #读取图片，输入到网络中，获得结果
    img = cv2.imread(imagePath)
    rst = predictor(img)
    _, path = save_result(i, img, rst)
    recognize(path)

    #关闭连接
    sock.close()
    print('Connection from %s:%s closed.' % addr)
    print("waiting for next connect")


def main(args):
    #检查输入参数
    if not os.path.exists(FLAGS.premodel):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(FLAGS.premodel))
    if not os.path.exists(FLAGS.cropPath):
        os.makedirs(FLAGS.cropPath)

    #获取测试图片路径列表
    pathList = getfilelist(FLAGS.imgPath)
    #获取前向传播函数
    predictor = getPredictor()
    #获取识别函数
    recognize = getRecognize() 
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 监听端口:
    s.bind(('0.0.0.0', 9998))
    s.listen(5)
    print('Waiting for connection...')

    while True:
        #接受一个新连接
        sock, addr = s.accept()
        #创建新线程来处理TCP连接
        t = threading.Thread(target = tcplink, args = (sock, addr, predictor, recognize))
        t.start()


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--premodel', 
                        type=str, 
                        help='The path of pretained model',
                        default='premodel')
    parser.add_argument('--imgPath', 
                        type=str, 
                        help='The path of testimage',
                        default='image')
    parser.add_argument('--cropPath', 
                        type=str,
                        help='where to save the croped image', 
                        default='result')
    parser.add_argument('--eastWeightsPath', 
                        type=str, 
                        help='Where you store the east weights',
                        default='premodel/model1/model.ckpt-122401')
    parser.add_argument('--crnnWeightsPath', 
                        type=str, 
                        help='Where you store the crnn weights',
                        default='premodel/model2/shadownet_2018-01-12-17-11-00.ckpt-159000')

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)

