/*
  ���ļ���ʵ����ͼ��Ļ�����������:
1����ȡĳλ�õ����ص�
2������ĳλ�õ����ص㣨8λ��32λ��64λ����
3����������֮��ľ����ƽ��
4����ͼƬĳһ�㻭һ����X��
5��������ͼƬ�ϳ�Ϊһ�������Ƕ���֮�ͣ����Ƕ��ߵĽϴ��ߡ�img1 �����Ͻǣ�img2�����½ǡ�
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