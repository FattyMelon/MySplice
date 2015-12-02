/*
  ���ļ�����Ҫ
  ����SIFT���������ʵ��
*/

#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"

#include "opencv\cxcore.h"
#include "opencv\cv.h"

/************************* Local Function Prototypes *************************/
static IplImage* create_init_img(IplImage*, int, double);
static IplImage* convert_to_gray32(IplImage*);


/*********************** Functions prototyped in sift.h **********************/

/*ʹ��Ĭ�ϲ�����ͼ������ȡSIFT������
������
img��ͼ��ָ��
feat�������洢�������feature�����ָ��
      ��������ڴ潫�ڱ������б����䣬ʹ���������ڵ��ó��ͷţ�free(*feat)
����ֵ����ȡ�������������������-1������ȡʧ��
*/
int sift_features(IplImage* img, struct feature** feat)
{
	//����_sift_features()����������������
	return _sift_features( img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
							SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
							SIFT_DESCR_HIST_BINS );
}

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

int _sift_features( IplImage* img, struct feature** feat, int intvls,
					double sigma,double contr_thr, int curv_thr, int img_dbl, 
					int descr_width, int descr_hist_bins )
{
	IplImage* init_img;	//��ʼ�����ͼ��
	IplImage*** gauss_pyr, *** dog_pyr;	//����ָ�룬��˹������ͼ���飬DoG������ͼ����
	CvMemStorage* storage; //�洢��
	CvSeq* features; //�洢����������У������д�ŵ���struct feature���͵�ָ��
	int octvs, i, n = 0;

	//����������
	if (!img) 
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}

	if (!feat)
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}

	//#����һ�������߶ȿռ䣬��������˹��֣�DoG��������dog_pyr
	//ת��Ϊ32λ�Ҷ�ͼ����һ����Ȼ�����һ�θ�˹ƽ��
	init_img = create_init_img(img, img_dbl, sigma);

	return n;
}


/************************ Functions prototyped here **************************/

/*��ԭͼת��Ϊ32λ�Ҷ�ͼ����һ����Ȼ�����һ�θ�˹ƽ���������ݲ���img_dbl�����Ƿ�ͼ��ߴ�Ŵ�Ϊԭͼ��2��
������
img�������ԭͼ��
img_dbl���Ƿ�ͼ��Ŵ�Ϊ֮ǰ������
sigma����ʼ��˹ƽ��������
����ֵ����ʼ����ɵ�ͼ��
*/
static IplImage* create_init_img(IplImage* img, int img_dbl, double sigma)
{
	IplImage* gray, * dbl;
	float sig_diff;

	//���ú�����������ͼ��ת��Ϊ32λ�Ҷ�ͼ������һ��
	//gray = convert_to_gray32(img);

	//�������˽�ͼ��Ŵ�Ϊԭͼ��2��
	if (img_dbl)
	{
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
		dbl = cvCreateImage(cvSize(img->width * 2, img->height * 2), IPL_DEPTH_32F, 1);   //�����Ŵ�ͼ��
		cvResize(gray, dbl, CV_INTER_CUBIC);  //�Ŵ�ԭͼ�ĳߴ磬����������ֵ

		//��˹ƽ������˹����x��y�����ϵı�׼���sig_diff
		cvSmooth(dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);
		cvReleaseImage(&gray);
		
		return dbl;
	}
	else
	{
		//�����0��ĳ߶�
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
		//��˹ƽ��
		cvSmooth(gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);

		return gray;
	}
}

/*������ͼ��ת��Ϊ32λ�Ҷ�ͼ,�����й�һ��
������
img������ͼ��3ͨ��8λ��ɫͼ(BGR)��8λ�Ҷ�ͼ
����ֵ��32λ�Ҷ�ͼ
*/
static IplImage* convert_to_gray32(IplImage* img)
{
	IplImage* gray8, * gray32;
	
	//����32λ��ͨ��ͼ��
	gray32 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	
	//���Ƚ�ԭͼת��Ϊ8λ��ͨ��ͼ��
	if (img->nChannels == 1) //��ԭͼ������ǵ�ͨ������ֱ�Ӹ���
		gray8 = cvClone(img);
	else //��ԭͼ����ͨ��ͼ��
	{
		gray8 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray8, CV_BGR2GRAY);  //��ԭͼת��Ϊ8λ��ͨ��ͼ��
	}

	//��8λ��ͨ��ͼ��ת��Ϊ32λ��ͨ��������һ��������255��
	cvConvertScale(gray8, gray32, 1.0 / 255.0, 0);

	cvReleaseImage(&gray8);

	return gray32; //����32λ��ͨ��ͼ��
}