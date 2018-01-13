data文件夹存放CRNN训练时及识别时使用的字符与序号映射表
image文件夹存放测试图像
lanms文件夹存放了用于优化RBOX结果的函数
model文件夹中存放了east和crnn网络的配置文件
premodel文件中，model1存放east的ckpt文件，model2存放crnn的ckpt文件
main1.py将east和crnn网络连接起来形成一个系统
server.py则是在main1.py中加入了socket通信
==================================
2018-1-13 对server进行更新：原本的server.py是将接受的图片进行定位并切片保存，之后重新读取这些切片进行识别。新版的server.py则去除了保存切片这一步，转而直接根据定位信息在原图上进行识别并进行相应的标定，然后保存结果图

