/*
  此文件中实现了图像的基本操作函数:
1、获取某位置的像素点
2、设置某位置的像素点（8位，32位和64位），
3、计算两点之间的距离的平方
4、在图片某一点画一个“X”
5、将两张图片合成为一个，高是二者之和，宽是二者的较大者。img1 在左上角，img2在右下角。
*/

#include "utils.h"

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"

#include <errno.h>
#include <string.h>
#include <stdlib.h>

/*************************** Function Definitions ****************************/
void fatal_error(char* format, ...)
{
	fprintf(stderr, "Error:");

	fprintf(stderr, format);

	fprintf(stderr, "\n");

	abort();
}