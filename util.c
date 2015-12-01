/*
  此文件中声明了图像的基本操作函数:
1、获取某位置的像素点
2、设置某位置的像素点（8位，32位和64位），
3、计算两点之间的距离的平方
4、在图片某一点画一个“X”
5、将两张图片合成为一个，高是二者之和，宽是二者的较大者。img1 在左上角，img2在右下角。
*/

#ifndef UTILS_H
#define UTILS_H

#include "opencv\cxcore.h"
#include <stdio.h>

/* 求x的绝对值
因为不确定返回的是int还是double
所以没有用inline函数 */
#ifndef ABS
#define ABS(x) (((x) > 0) ? (-x) : (x)) 
#endif


#endif