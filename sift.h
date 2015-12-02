/*
  ���ļ��а���SIFT�������⺯�����������Լ�һЩʵ��SIFT�㷨��һЩĬ�ϲ����Ķ���
*/

#ifndef SIFT_H
#define SIFT_H

#include "opencv\cxcore.h"

/******************************** Structures *********************************/

//��ֵ�������õ��Ľṹ
//��SIFT������ȡ�����У����������ݻᱻ��ֵ��feature�ṹ��feature_data��Ա
struct detection_data
{
    int r;      //���������ڵ���
    int c;      //���������ڵ���
    int octv;   //��˹��ֽ������У����������ڵ���
    int intvl;  //��˹��ֽ������У����������ڵ����еĲ�
    double subintvl;  //�������ڲ㷽��(�ҷ���,intvl����)�ϵ�������ƫ����
    double scl_octv;  //���������ڵ���ĳ߶�
};

struct feature;


/******************************* һЩĬ�ϲ��� *****************************/

//��˹������ÿ���ڵĲ���
/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

//��0��ĳ�ʼ�߶ȣ�����0���˹ģ����ʹ�õĲ���
/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6

//�Աȶ���ֵ����Թ�һ�����ͼ������ȥ�����ȶ�����
/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

//�����ʱ�ֵ����ֵ������ȥ����Ե����
/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

//�Ƿ�ͼ��Ŵ�Ϊ֮ǰ������
/** double image size before pyramid construction? */
#define SIFT_IMG_DBL 1

//����ͼ��ĳ߶�Ϊ0.5
/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

//�߽�����ؿ��ȣ��������н����Ա߽����еļ�ֵ�㣬��ֻ���߽��������Ƿ���ڼ�ֵ��
/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

//ͨ����ֵ���м�ֵ�㾫ȷ��λʱ������ֵ���������ؼ�����������
/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

//�����㷽��ֵ�����У��ݶȷ���ֱ��ͼ������(bin)�ĸ���
/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

//�����㷽��ֵ�����У���������İ뾶Ϊ��3 * 1.5 * ��
/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

//�����㷽��ֵ�����У���������İ뾶Ϊ��3 * 1.5 * ��
/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

//�����㷽��ֵ�����У��ݶȷ���ֱ��ͼ��ƽ��������������ݶ�ֱ��ͼ��Ҫ���и�˹ƽ��
/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

//�����㷽��ֵ�����У��ݶȷ�ֵ�ﵽ���ֵ��80%�����Ϊ����������
/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

//�������������ӹ����У����㷽��ֱ��ͼʱ���������㸽������Ϊd*d������ÿ����������һ��ֱ��ͼ��SIFT_DESCR_WIDTH��d��Ĭ��ֵ
/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

//�������������ӹ����У�ÿ������ֱ��ͼ��bin����
/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

//�������������ӹ����У���������Χ��d*d�������У�ÿ������Ŀ���Ϊm*�Ҹ����أ�SIFT_DESCR_SCL_FCTR��m��Ĭ��ֵ����Ϊ������ĳ߶�
/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

//�������������ӹ����У�����������������Ԫ�ص���ֵ(���ֵ����������Թ�һ���������������)����������ֵ��Ԫ�ر�ǿ�и�ֵΪ����ֵ
/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

//�������������ӹ����У��������͵����������ӱ�Ϊ����ʱ���Ե�ϵ��
/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

//������һ���������ĺ����꣬������ȡ����f�е�feature_data��Ա��ת��Ϊdetection_data��ʽ��ָ��
/* returns a feature's detection data */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )


/*************************** Function Prototypes *****************************/

/*ʹ��Ĭ�ϲ�����ͼ������ȡSIFT������
������
img��ͼ��ָ��
feat�������洢�������feature�����ָ��
      ��������ڴ潫�ڱ������б����䣬ʹ���������ڵ��ó��ͷţ�free(*feat)
����ֵ����ȡ�������������������-1������ȡʧ��
*/
extern int sift_features( IplImage* img, struct feature** feat );


/*ʹ���û�ָ���Ĳ�����ͼ������ȡSIFT������
������
img������ͼ��
feat���洢������������ָ��
      ��������ڴ潫�ڱ������б����䣬ʹ���������ڵ��ó��ͷţ�free(*feat)
intvls��ÿ��Ĳ���
sigma����ʼ��˹ƽ��������
contr_thr���Աȶ���ֵ����Թ�һ�����ͼ������ȥ�����ȶ�����
curv_thr��ȥ����Ե����������������ֵ
img_dbl���Ƿ�ͼ��Ŵ�Ϊ֮ǰ������
descr_width���������������У����㷽��ֱ��ͼʱ���������㸽������Ϊdescr_width*descr_width������ÿ����������һ��ֱ��ͼ
descr_hist_bins���������������У�ÿ��ֱ��ͼ��bin�ĸ���
����ֵ����ȡ�������������������-1������ȡʧ��
*/
extern int _sift_features( IplImage* img, struct feature** feat, int intvls,
						  double sigma, double contr_thr, int curv_thr,
						  int img_dbl, int descr_width, int descr_hist_bins );


#endif