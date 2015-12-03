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

/***************************** Inline Functions ******************************/

//从8位图中获取像素点
static __inline int pixval8(IplImage* img, int r, int c)
{
	return (int)(((uchar*)(img->imageData + img->widthStep*r))[c]);
}

//设置8位图像素点
static __inline void setpix8(IplImage* img, int r, int c, uchar val)
{
	((uchar*)(img->imageData + img->widthStep*r))[c] = val;
}

/*从32位浮点型单通道图像中获取指定坐标的像素值，内联函数
参数：
img：输入图像
r：行坐标
c：列坐标
返回值：坐标(c,r)处(r行c列)的像素值
*/
static __inline float pixval32f(IplImage* img, int r, int c)
{
	return ((float*)(img->imageData + img->widthStep * r))[c];
}

//设置32位图的像素点
static __inline void setpix32f(IplImage* img, int r, int c, float val)
{
	((float*)(img->imageData + img->widthStep * r))[c] = val;
}

/**************************** Function Prototypes ****************************/

//错误处理
/**
Prints an error message and aborts the program.  The error message is
of the form "Error: ...", where the ... is specified by the \a format
argument

@param format an error message format string (as with \c printf(3)).
*/
extern void fatal_error( char* format, ... );

#endif