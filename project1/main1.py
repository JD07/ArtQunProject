__author__ = 'JD07'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import collections
import functools
import argparse
import datetime
import logging
import uuid
import json
import time
import sys
import cv2
import os
import re

from local_utils import  data_utils
from global_configuration import config
from icdar import restore_rectangle
from crnn_model import crnn_model

import model
import lanms


FLAGS = None

def getfilelist(path):
    #通过该函数遍历文件夹下的jpg等
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
    '''
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)#按顺序两个元素组成一列，形成四个点的格式
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def cropImg(dirPath, img, rst):
    for i,t in enumerate(rst['text_lines']):
        x0 = int(min(t['x0'], t['x1'], t['x2'], t['x3']))
        x1 = int(max(t['x0'], t['x1'], t['x2'], t['x3']))
        y0 = int(min(t['y0'], t['y1'], t['y2'], t['y3']))
        y1 = int(max(t['y0'], t['y1'], t['y2'], t['y3']))
        rstImg = img[y0:y1, x0:x1]
        rstName = str(i) + '.jpg'
        rstPath = os.path.join(dirPath, rstName)
        cv2.imwrite(rstPath, rstImg)

def save_result(i, img, rst):
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
    恢复RBOX框
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
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
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


def load_model(sess, model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(sess, os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    '''
        获取指定路径下meta文件和ckpt文件的路径
        注意：该函数存在问题，只有特定格式保存的网络可以获取准确的ckpt_file，需要后续修改
    '''
    ckpt_file = 'model.ckpt-13975' #函数存在问题，ckpt暂时只能通过手动赋值来保证正常运行
    files = os.listdir(model_dir)

    #获取meta文件名
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    
    #获取ckpt文件名（目前存在问题）
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]

    return meta_file, ckpt_file

def getPredictor():
    with tf.Graph().as_default() as net1_graph:
        #该函数用于从ckpt文件中恢复网络，并返回一个可以得到相应图像结果的predictor函数
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
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
    with tf.Graph().as_default() as net2_graph:
        inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')

        net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=19)

        with tf.variable_scope('shadow'):
            net_out = net.build_shadownet(inputdata=inputdata)

        decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)

        decoder = data_utils.TextFeatureIO()

        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

        # config tf saver
        saver2 = tf.train.Saver()

    sess2 = tf.Session(graph=net2_graph, config=sess_config)   
    saver2.restore(sess=sess2, save_path=FLAGS.crnnWeightsPath)

    def recognize(path):
        imageList = getfilelist(path)
        for imagePath in imageList:
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 32))
            image = np.expand_dims(image, axis=0).astype(np.float32)
            preds = sess2.run(decodes, feed_dict={inputdata: image})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])
            print('Predict image {:s} label {:s}'.format(os.path.split(imagePath)[1], preds[0]))
    
    return recognize


def main(args):
    #检查输入参数
    if not os.path.exists(FLAGS.premodel):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(FLAGS.premodel))
    if not os.path.exists(FLAGS.imgPath):
        raise RuntimeError(
            'ImgPath `{}` not found'.format(FLAGS.imgPath))
    if not os.path.exists(FLAGS.cropPath):
        os.makedirs(FLAGS.cropPath)

    #获取测试图片路径列表
    pathList = getfilelist(FLAGS.imgPath)
    #获取前向传播函数
    predictor = getPredictor()
    #获取识别函数
    recognize = getRecognize() 
    
    #path='result/0'
    #recognize(path)
    
    #开始循环
    for i,imgPath in enumerate(pathList):
        img = cv2.imread(imgPath)
        rst = predictor(img)
        _, path = save_result(i, img, rst)
        recognize(path)
    


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
                        default='premodel/model.ckpt-13975')
    parser.add_argument('--crnnWeightsPath', 
                        type=str, 
                        help='Where you store the crnn weights',
                        default='crnnModel/myShadownet/shadownet_2017-12-03-23-20.ckpt-31688')

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
