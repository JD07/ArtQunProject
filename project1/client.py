__author__ = 'JD07'

import socket,time
import os
import random


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 建立连接:
    s.connect(('127.0.0.1', 9998))
    
    #发送图像路劲
    #index = random.randrange(0,10)
    index = input()
    imgName = index + '.jpg'
    imgName = os.path.join('image', imgName) + "$ahfjkafka"
    s.send(imgName.encode())
    s.close()
