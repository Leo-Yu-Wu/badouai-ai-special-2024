import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
————————————————————————————————————————————————————
致审阅老师：
真的非常抱歉，实在是最近时间有点紧张，从PCA那期开始作业交的慢了很多
这些作业我是期望能够把全过程了解透彻，但苦于不是开发人员，虽然自学了python在代码上没啥问题，但一结合算法就有时候要看很久代码是什么意思
包括每个阶段的算法是怎么用代码实现的，您写的和网上自己查阅的资料有什么区别，还有哪些其他的额外知识需要补充
这一下给我本来不富裕的时间弄的更紧张了
————————————————————————————————————————————————————
canny这期和后面的我先上交简化版的作业，后续我会尽量把我自己理解的内容做成py文件上传到git上，像之前三期作业那样
麻烦审阅老师了！！！
————————————————————————————————————————————————————
'''

'''
canny需要由五部实现

1、去噪声：应用高斯滤波来平滑图像，目的是去除噪声
2、梯度：通过边缘幅度的算法，找寻图像的梯度
3、非极大值抑制：应用非最大抑制技术来过滤掉非边缘像素，将模糊的边界变得清晰。该过程保留了每个像素点上梯度强度的极大值，过滤掉其他的值。
4、应用双阈值的方法来决定可能的（潜在的）边界；
5、利用滞后技术来跟踪边界。若某一像素位置和强边界相连的弱边界认为是边界，其他的弱边界则被删除。
'''


'''
第一部分
关于去噪声，或者叫图像平滑
目前学的有以下几种图像平滑的滤波器

1、均值滤波
均值滤波是滤波器N*N的中心点，改为滤波器所有像素的平均值的方法。
即：滤波器像素值的和/滤波器像素数量【可以是3*3、5*5、7*7……】
可调用cv2中的blur方法实现，其中核大小通过元祖的方式传递，不能直接写一个值，所以可以写非正方形的滤波器
目前来看效果还是可以的

2、中值滤波
中值滤波是将滤波器N*N内所有像素按大小排序，取中值的方法。
例如：5*5的滤波器，有25个值，则取第13个值为滤波器的值
可调用cv2中的medianBlur方法实现，核大小可以直接用一个数实现
目前来看效果比较好

3、高斯滤波
高斯滤波是通过将每个滤波器对应位置的像素，与滤波器中的值进行乘积，最终在求和计算的方法，目前可知的有3*3、5*5的滤波器
例如：3*3的滤波器中，用原像素值p1*滤波器系数x1 + 原像素值p2*滤波器系数x2 + 原像素值p3*滤波器系数x3 + …… + 原像素值p9*滤波器系数x9
可调用cv2中的GaussianBlur方法实现，核大小需要用元祖传递，还需要传递一个方差的值
目前来看对图像的影响最小的
'''




if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('lenna.png'), cv2.COLOR_BGR2GRAY)
    cv2.imshow("canny", cv2.Canny(img, 200, 300))
    cv2.waitKey(0)






















