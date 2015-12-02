/*
  ���ļ�����Ҫ
  ����SIFT���������ʵ��
*/

#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"

#include "opencv\cv.h"
#include "opencv\cxcore.h"


/************************* Local Function Prototypes *************************/
static IplImage* create_init_img(IplImage*, int, double);
static IplImage* convert_to_gray32(IplImage*);
static IplImage*** build_gauss_pyr(IplImage*, int, int, double);
static IplImage* downsample(IplImage*);
static IplImage*** build_dog_pyr(IplImage*, int, int);


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

	/*if (!feat)
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}*/

	//#����һ�������߶ȿռ䣬��������˹��֣�DoG��������dog_pyr
	//ת��Ϊ32λ�Ҷ�ͼ����һ����Ȼ�����һ�θ�˹ƽ��
	init_img = create_init_img(img, img_dbl, sigma);
	//�����˹������������octvs
	octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;
	//��ÿһ��Ķ����ø�˹ģ������3��ͼ�����Ը�˹������ÿ����intvls+3�㣬DOG������ÿ����intvls+2��
	//������˹������gauss_pyr,��һ��octvs*��intvls+3����ͼ������
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);

	//������˹��֣�DoG��������dog_pyr
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);

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
	gray = convert_to_gray32(img);

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

/*�����������������˹������
������
base������ͼ����Ϊ��˹�������Ļ�ͼ��
octvs����˹������������
intvls��ÿ��Ĳ���
sigma����ʼ�߶�
����ֵ����˹����������һ��octvs*(intvls+3)��ͼ������
*/
static IplImage*** build_gauss_pyr(IplImage* base, int octvs, int intvls, double sigma)
{
	IplImage*** gauss_pyr;
	
	double* sig = calloc(intvls + 3, sizeof(double)); //ÿ���sigma��������
	double sig_total, sig_prev, k;
	int i, o;

	//Ϊ��˹������gauss_pyr����ռ䣬��octvs��Ԫ�أ�ÿ��Ԫ����һ��ͼ�����ָ��
	gauss_pyr = calloc(octvs, sizeof(IplImage**));
	//Ϊ��i��ͼ�����ռ䣬��intvls+3��Ԫ��
	for (i = 0; i< octvs; ++i) 
		gauss_pyr[i] = calloc(intvls + 3, sizeof(IplImage* ));

	/*	����sigma�Ĺ�ʽ
		sigma_{total}^2 = sigma_{i}^2 + sigma_{i-1}^2   */
	sig[0] = sigma;	//���Գ߶�
	k = pow(2.0, 1.0 / intvls);

	for (i = 1; i < intvls +3; ++i) 
	{
		sig_prev = pow(k, i - 1) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	//ÿ��ÿ�����ɸ�˹������
	for (o = 0; o < octvs; ++o) 
		for( i = 0; i < intvls + 3; ++i) 
		{
			if (o == 0 && i == 0)	//��0�飬��0�㣬����ԭͼ��
				gauss_pyr[o][i] = cvCloneImage(base);
			else if (i == 0)	//�µ�һ����ײ�ͼ��������һ�����һ��ͼ�����²����õ���
				gauss_pyr[o][i] = downsample(gauss_pyr[o-1][intvls]);
			else	//���ϲ�ͼ����и�˹ƽ���õ���ǰ��ͼ��
			{
				//����ͼ��
				gauss_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i-1]), IPL_DEPTH_32F, 1);
				//����һ��ͼ��gauss_pyr[o][i-1]���в���Ϊsig[i]�ĸ�˹ƽ��
				cvSmooth(gauss_pyr[o][i-1], gauss_pyr[o][i], CV_GAUSSIAN, 0, 0, sig[i], sig[i]);
			}
		}

	free(sig);	//�ͷ�sigma��������

	return gauss_pyr;
}

/*������ͼ�����²����������ķ�֮һ��С��ͼ��(ÿ��ά���ϼ���)��ʹ������ڲ�ֵ����
������
img������ͼ��
����ֵ���²������ͼ��
*/
static IplImage* downsample(IplImage* img)
{
	//�²���ͼ��
	IplImage* smaller = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);
	cvResize(img, smaller, CV_INTER_NN);	//����ڲ�ֵ
	
	return smaller;
}

/*ͨ���Ը�˹��������ÿ��������ͼ�������������˹��ֽ�����
������
gauss_pyr����˹������
octvs������
intvls��ÿ��Ĳ���
����ֵ����˹��ֽ���������һ��octvs*(intvls+2)��ͼ������
*/
static IplImage*** build_dog_pyr(IplImage*** gauss_pyr, int octvs, int intvls)
{
	IplImage*** dog_pyr;
	int i, o;

	//Ϊ��˹��ֽ���������ռ䣬��octvs��Ԫ�أ�ÿ��Ԫ����һ��ͼ�����ָ��
	dog_pyr = calloc(octvs, sizeof(IplImage**));
	//Ϊ��i��ͼ��dog_pyr[i]����ռ䣬��intvls+2��Ԫ��
	for (i = 0; i < octvs; ++i)
	{
		dog_pyr[i] = calloc(intvls + 2, sizeof(IplImage*));
	}

	//ÿ��ÿ�������ͼ��
	for (o = 0; o < octvs; ++o)
		for (i = 0; i < intvls + 2; ++i)
		{
			//����DoG�������ĵ�o���i��Ĳ��ͼ��
			dog_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i]), IPL_DEPTH_32F, 1);
			//����˹�������ĵ�o���i+1��ͼ���ȥ��i��ͼ��
			cvSub(gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL);
		}

	return dog_pyr;	//���ظ�˹��ֽ�����
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