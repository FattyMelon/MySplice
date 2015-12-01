/*
  ���ļ�����洢������Ľṹ��feature���Լ���������ԭ�͵�������
1��������ĵ���͵���
2�����������
*/

#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "opencv\cxcore.h"

/*���������ͣ�
FEATURE_OXFD��ʾţ���ѧVGG�ṩ��Դ���е��������ʽ
FEATURE_LOWE��ʾDavid.Lowe�ṩ��Դ���е��������ʽ
*/
enum feature_type {
	FEATURE_OXFD,
	FEATURE_LOWE
};

/*������ƥ�����ͣ�
FEATURE_FWD_MATCH������feature�ṹ�е�fwd_match���Ƕ�Ӧ��ƥ���
FEATURE_BCK_MATCH������feature�ṹ�е�bck_match���Ƕ�Ӧ��ƥ���
FEATURE_MDL_MATCH������feature�ṹ�е�mdl_match���Ƕ�Ӧ��ƥ���
*/
enum feature_match_type
{
	FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH,
	FEATURE_MDL_MATCH
};

/*���������������ɫ*/
#define FEATURE_OXFD_COLOR CV_RGB(255,255,0)
#define FEATURE_LOWE_COLOR CV_RGB(255,0,255)

/*������������ӳ��ȣ���Ϊ128*/
#define FEATURE_MAX_D 128

/*������ṹ��
�˽ṹ��ɴ洢2�����͵������㣺
FEATURE_OXFD��ʾ��ţ���ѧVGG�ṩ��Դ���е��������ʽ��
FEATURE_LOWE��ʾ��David.Lowe�ṩ��Դ���е��������ʽ��
�����OXFD���͵������㣬�ṹ���е�a,b,c��Ա��������������Χ�ķ�������(��Բ�Ĳ���)��������
�����LOWE���͵������㣬�ṹ���е�scl��ori��Ա������������Ĵ�С�ͷ���
fwd_match��bck_match��mdl_matchһ��ͬʱֻ��һ�������ã�����ָ�����������Ӧ��ƥ���
*/
struct feature
{
	double x;						//������x������
	double y;						//������y������
	double a;
	double b;
	double c;
	double scl;						//LOWE������ĳ߶�
	double ori;						//LOWE������ķ���
	int d;							//���������ӵ�ά����һ��Ϊ128
	double descr[FEATURE_MAX_D];	//128ά������������
	int type;						//����������ͣ�OXFD��LOWE
	int category;					//�����������
	struct feature* fwd_match;		//
	struct feature* bck_match;
	struct feature* mdl_match;
	CvPoint2D64f img_pt;			//�����������
	CvPoint2D64f mdl_pt;			//ƥ������Ϊmdl_matchʱ�������
	void* feature_data;				//�û����������:
									//��SIFT��ֵ�����У���detection_data�ṹ��ָ��
									//��k-d�������У���bbf_data�ṹ��ָ��
									//��RANSAC�㷨�У���ransac_data�ṹ��ָ��
};

/*���ļ��ж���ͼ������
�ļ��е��������ʽ������FEATURE_OXFD��FEATURE_LOWE��ʽ
������
filename���ļ���
type������������
feat�������洢�������feature�����ָ��,��ά����
����ֵ����������������
*/
extern int import_features(char* filename, int type, struct feature** feature);


#endif