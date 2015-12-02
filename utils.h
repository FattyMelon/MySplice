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