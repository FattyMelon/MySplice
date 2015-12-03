/*
  ���ļ���������ͼ��Ļ�����������:
1����ȡĳλ�õ����ص�
2������ĳλ�õ����ص㣨8λ��32λ��64λ����
3����������֮��ľ����ƽ��
4����ͼƬĳһ�㻭һ����X��
5��������ͼƬ�ϳ�Ϊһ�������Ƕ���֮�ͣ����Ƕ��ߵĽϴ��ߡ�img1 �����Ͻǣ�img2�����½ǡ�
*/

#ifndef UTILS_H
#define UTILS_H

#include "opencv\cxcore.h"
#include <stdio.h>

/* ��x�ľ���ֵ
��Ϊ��ȷ�����ص���int����double
����û����inline���� */
#ifndef ABS
#define ABS(x) (((x) > 0) ? (-x) : (x)) 
#endif

/***************************** Inline Functions ******************************/

//��8λͼ�л�ȡ���ص�
static __inline int pixval8(IplImage* img, int r, int c)
{
	return (int)(((uchar*)(img->imageData + img->widthStep*r))[c]);
}

//����8λͼ���ص�
static __inline void setpix8(IplImage* img, int r, int c, uchar val)
{
	((uchar*)(img->imageData + img->widthStep*r))[c] = val;
}

/*��32λ�����͵�ͨ��ͼ���л�ȡָ�����������ֵ����������
������
img������ͼ��
r��������
c��������
����ֵ������(c,r)��(r��c��)������ֵ
*/
static __inline float pixval32f(IplImage* img, int r, int c)
{
	return ((float*)(img->imageData + img->widthStep * r))[c];
}

//����32λͼ�����ص�
static __inline void setpix32f(IplImage* img, int r, int c, float val)
{
	((float*)(img->imageData + img->widthStep * r))[c] = val;
}

/**************************** Function Prototypes ****************************/

//������
/**
Prints an error message and aborts the program.  The error message is
of the form "Error: ...", where the ... is specified by the \a format
argument

@param format an error message format string (as with \c printf(3)).
*/
extern void fatal_error( char* format, ... );

#endif