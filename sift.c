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
static IplImage*** build_dog_pyr(IplImage***, int, int);
static CvSeq* scale_space_extrema(IplImage***, int, int, double, int, CvMemStorage*);
static int is_extremum(IplImage***, int, int, int, int);
static struct feature* interp_extremum(IplImage***, int, int, int, int, int, double);
static void interp_step(IplImage***, int , int, int, int, double*, double*, double*);
static CvMat* deriv_3D(IplImage***, int, int, int, int);
static CvMat* hessian_3D(IplImage***, int, int, int, int);
static double interp_contr(IplImage***, int, int, int, int, double, double, double);
static struct feature* new_feature( void);
static int is_too_edge_like(IplImage*, int, int, int);


//static void calc_feature_scales( CvSeq*, double, int );

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

	//����һ�������߶ȿռ䣬��������˹��֣�DoG��������dog_pyr
	//ת��Ϊ32λ�Ҷ�ͼ����һ����Ȼ�����һ�θ�˹ƽ��
	init_img = create_init_img(img, img_dbl, sigma);
	//�����˹������������octvs
	octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;
	//��ÿһ��Ķ����ø�˹ģ������3��ͼ�����Ը�˹������ÿ����intvls+3�㣬DOG������ÿ����intvls+2��
	//������˹������gauss_pyr,��һ��octvs*��intvls+3����ͼ������
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);

	//������˹��֣�DoG��������dog_pyr
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);

	//��������ڳ߶ȿռ��м�⼫ֵ�㣬�����о�ȷ��λ��ɸѡ
	//����Ĭ�ϴ�С���ڴ�洢��
	storage = cvCreateMemStorage(0);
	//�ڳ߶ȿռ��м�⼫ֵ��
	features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, storage);
	//��������������features��ÿ��������ĳ߶�
	//calc_feature_scales(features, sigma, intvls);

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

/*�ڳ߶ȿռ��м�⼫ֵ�㣬ͨ����ֵ��ȷ��λ��ȥ���ͶԱȶȵĵ㣬ȥ����Ե�㣬���ؼ�⵽������������
������
dog_pyr����˹��ֽ�����
octvs����˹��ֽ�����������
intvls��ÿ��Ĳ���
contr_thr���Աȶ���ֵ����Թ�һ�����ͼ������ȥ�����ȶ�����
cur_thr�������ʱ�ֵ����ֵ������ȥ����Ե����
storage���洢��
����ֵ�����ؼ�⵽�������������
*/
static CvSeq* scale_space_extrema(IplImage*** dog_pyr, int octvs, int intvls, 
									double contr_thr, int curv_thr, CvMemStorage* storage)
{
	CvSeq* features;	//����������
	double prelim_contr_thr = 0.5 * contr_thr / intvls;	//���ضԱȶȵ���ֵ
	struct feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;

	//�ڴ洢��storage�ϴ����洢��ֵ������У����д洢feature�ṹ���͵�����
	features = cvCreateSeq(0, sizeof(CvSeq), sizeof(struct feature), storage);

	//������˹��ֽ���������⼫ֵ��
	//���Ա߽��ߣ�ֻ���߽������ڵļ�ֵ��
	for (o = 0; o < octvs; ++o)
		for (i = 1; i <= intvls; ++i)
			for (r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; ++r)	//��r��
				for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; ++c)	//��c��
					//���г����ĶԱȶȼ�飬ֻ�е���һ���������ֵ���ڶԱȶ���ֵprelim_contr_thrʱ�ż����������ص��Ƿ�����Ǽ�ֵ
					if (ABS(pixval32f(dog_pyr[o][i], r, c)) > prelim_contr_thr)
					{
						//ͨ���ڳ߶ȿռ��н�һ�����ص��ֵ������Χ3*3*3�����ڵĵ�Ƚ��������˵��Ƿ�ֵ��(����ֵ��С����)
						if (is_extremum(dog_pyr, o, i, r, c))	//���Ǽ�ֵ��
						{
							//���ڼ�ֵ��ļ��������ɢ�ռ��н��еģ����Լ�⵽�ļ�ֵ�㲢��һ�������������ϵļ�ֵ��
							//��Ϊ�����ļ�ֵ�����λ����������֮�䣬������ɢ�ռ���ֻ�ܾ�ȷ������㾫����
							//ͨ�������ؼ���ֵ���м�ֵ�㾫ȷ��λ(������ֵ������)����ȥ���ͶԱȶȵļ�ֵ�㣬������������������feature�ṹ����
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);

							if (feat)
							{
								ddata = feat_detection_data(feat);
								//if(!is_too)
							}
						}
					}
}

/*ͨ���ڳ߶ȿռ��н�һ�����ص��ֵ������Χ3*3*3�����ڵĵ�Ƚ��������˵��Ƿ�ֵ��(����ֵ��С����)
������
dog_pyr����˹��ֽ�����
octv�����ص����ڵ���
intvl�����ص����ڵĲ�
r�����ص����ڵ���
c�����ص����ڵ���
����ֵ����ָ�������ص��Ǽ�ֵ��(���ֵ������Сֵ)������1�����򷵻�0
*/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	float val = pixval32f(dog_pyr[octv][intvl], r, c);
	int i, j, k;

	//����Ƿ����ֵ
	if (val > 0)
	{
		for (i = -1; i <= 1; ++i)
			for (j = -1; j <= 1; ++j)
				for (k = -1; k <= 1; ++k)
					if (val < pixval32f(dog_pyr[octv][intvl+i], r + j, c + k))
						return 0;
	}
	//����Ƿ���Сֵ
	else
	{
		for( i = -1; i <= 1; ++i)//��
			for( j = -1; j <= 1; ++j )//��
				for( k = -1; k <= 1; ++k )//��
					if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k))
						return 0;
	}
}

/*ͨ�������ؼ���ֵ���м�ֵ�㾫ȷ��λ(������ֵ������)����ȥ���ͶԱȶȵļ�ֵ�㣬������������������feature�ṹ����
������
dog_pyr����˹��ֽ�����
octv�����ص����ڵ���
intvl�����ص����ڵĲ�
r�����ص����ڵ���
c�����ص����ڵ���
intvls��ÿ��Ĳ���
contr_thr���Աȶ���ֵ����Թ�һ�����ͼ������ȥ�����ȶ�����
����ֵ�����ؾ���ֵ�������������(feature����)���������޴β�ֵ��Ȼ�޷���ȷ������������߸õ�Աȶȹ��ͣ�����NULL
*/
static struct feature* interp_extremum(IplImage*** dog_pyr, int otcv, int intvl, 
										int r, int c, int intvls, double contr_thr)
{
	struct feature* feat;	//�������������
	struct detection_data * ddata;	//����������йصĽṹ������feature�ṹ��feature_data��Ա��
	double xi, xr, xc, contr;
	int i = 0;	//��ֵ����

	//
	while (i < SIFT_MAX_INTERP_STEPS)
	{
		//����һ�μ�ֵ���ֵ�������(�㷽��,intvl����)��y��x�����ϵ�������ƫ����(����)
		interp_step(dog_pyr, otcv, intvl, r, c, &xi, &xr, &xc);
		//�������ⷽ���ϵ�ƫ��������0.5����ζ�Ų�ֵ�����Ѿ�ƫ�Ƶ������ٽ����ϣ����Ա���ı䵱ǰ�ؼ����λ������
		if (ABS(xi) < 0.5 && ABS(xr) < 0.5 && ABS(xc) < 0.5)	//������������ѭ��
			break;

		//�����ؼ��������
		c += cvRound(xc);
		r += cvRound(xr);
		intvl += cvRound(xi);

		//�����������󳬳���Χ���������ֵ������NULL
		if (intvl < 1 || intvl > intvls ||	//������Խ��
			c < SIFT_IMG_BORDER ||	c >= dog_pyr[otcv][0]->width - SIFT_IMG_BORDER ||	//�����굽�߽�����
			r < SIFT_IMG_BORDER ||	r >= dog_pyr[otcv][0]->height - SIFT_IMG_BORDER)
		{
			return NULL;
		}

		++i;
	}

	if (i >= SIFT_MAX_INTERP_STEPS)
		return NULL;

	//���㱻��ֵ��ĶԱȶ�:D + 0.5 * dD^T * X
	contr = interp_contr(dog_pyr, otcv, intvl, r, c, xi, xr, xc);
	if (ABS(contr) < contr_thr / intvls)	//���õ�Աȶ�̫С������
		return NULL;

	//Ϊһ��������feature�ṹ����ռ䲢��ʼ��������������ָ��
	feat = new_feature();
	ddata = feat_detection_data(feat);

	//������������긳ֵ��������feat
	feat->img_pt.x = feat->x = (c + xc) * pow(2.0, otcv);
	feat->img_pt.y = feat->y = (r + xr) * pow(2.0, otcv);

	ddata->r = r;
	ddata->c = c;
	ddata->octv = otcv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;	//�������ڲ㷽��(�ҷ���,intvl����)�ϵ�������ƫ����

	return feat;
}

/*����һ�μ�ֵ���ֵ������x��y���ҷ���(�㷽��)�ϵ�������ƫ����(����)
	������
	dog_pyr����˹��ֽ�����
	octv�����ص����ڵ���
	intvl�����ص����ڵĲ�
	r�����ص����ڵ���
	c�����ص����ڵ���
	xi������������㷽���ϵ�����������(ƫ��)
	xr�����������y�����ϵ�����������(ƫ��)
	xc�����������x�����ϵ�����������(ƫ��)
*/
static void interp_step(IplImage*** dog_pyr, int octv, int intvl, 
						int r, int c, double* xi, double* xr, double* xc)
{
	CvMat* dD, * H, * H_inv, X;
	double x[3] = {0};

	//��DoG�������м���ĳ���x����y�����Լ��߶ȷ����ƫ����
	dD = deriv_3D(dog_pyr, octv, intvl, r, c);

	//��DoG�������м���ĳ���3*3��ɭ����
	H = hessian_3D(dog_pyr, octv, intvl, r, c);
	H_inv = cvCreateMat(3, 3, CV_64FC1);
	//�������
	cvInvert(H, H_inv, CV_SVD);
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	//X = - H^(-1) * dD��H������Ԫ�طֱ���x,y,�ҷ����ϵ�ƫ����
	cvGEMM(H_inv, dD, -1, NULL, 0, &X, 0);

	cvReleaseMat(&dD);	
	cvReleaseMat(&H);
	cvReleaseMat(&H_inv);

	*xi = x[2];	//�㷽���ƫ����
	*xr = x[1];
	*xc = x[0];
}

/*��DoG�������м���ĳ���x����y�����Լ��߶ȷ����ϵ�ƫ����
������
dog_pyr����˹��ֽ�����
octv�����ص����ڵ���
intvl�����ص����ڵĲ�
r�����ص����ڵ���
c�����ص����ڵ���
����ֵ������3��ƫ������ɵ�������{ dI/dx, dI/dy, dI/ds }^T
*/
static CvMat* deriv_3D(IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	CvMat* dI;
	double dx, dy, ds;

	//����������ƫ��,�������õĸ������ȡ��ֵ���ݶȼ��㷽��
	//��x�����ϵĲ�������ƴ���ƫ����
	dx = (pixval32f(dog_pyr[octv][intvl], r, c+1) -
		pixval32f(dog_pyr[octv][intvl], r, c-1)) / 2.0;
	//��y�����ϵĲ�������ƴ���ƫ����
	dy = ( pixval32f(dog_pyr[octv][intvl], r+1, c) -
		pixval32f(dog_pyr[octv][intvl], r-1, c)) / 2.0;
	//����Ĳ�������ƴ���߶ȷ����ϵ�ƫ����
	ds = ( pixval32f(dog_pyr[octv][intvl+1], r, c) -
		pixval32f(dog_pyr[octv][intvl-1], r, c )) / 2.0;

	//���������
	dI = cvCreateMat(3, 1, CV_64FC1);
	cvmSet(dI, 0, 0, dx);
	cvmSet(dI, 1, 0, dy);
	cvmSet(dI, 2, 0, ds);

	return dI;
}

/*��DoG�������м���ĳ���3*3��ɭ����
    / Ixx  Ixy  Ixs \
    | Ixy  Iyy  Iys |
    \ Ixs  Iys  Iss /
������
dog_pyr����˹��ֽ�����
octv�����ص����ڵ���
intvl�����ص����ڵĲ�
r�����ص����ڵ���
c�����ص����ڵ���
����ֵ������3*3�ĺ�ɭ����
*/
static CvMat* hessian_3D(IplImage*** dog_pyr, int octv, int intvl,
							int r, int c)
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f(dog_pyr[octv][intvl], r, c);	//�õ������ֵ

	//�ò�ֽ��ƴ��浹��
	//dxx = f(i+1,j) - 2f(i,j) + f(i-1,j)
	//dyy = f(i,j+1) - 2f(i,j) + f(i,j-1)
	dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
		pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
	dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
		pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
	dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
		pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
	dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
		pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
		pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
		pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
	dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
		pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
		pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
		pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
	dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
		pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
		pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
		pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;

	//��ɺ�ɭ����
	H = cvCreateMat( 3, 3, CV_64FC1 );
	cvmSet( H, 0, 0, dxx );
	cvmSet( H, 0, 1, dxy );
	cvmSet( H, 0, 2, dxs );
	cvmSet( H, 1, 0, dxy );
	cvmSet( H, 1, 1, dyy );
	cvmSet( H, 1, 2, dys );
	cvmSet( H, 2, 0, dxs );
	cvmSet( H, 2, 1, dys );
	cvmSet( H, 2, 2, dss );

	return H;
}

/*���㱻��ֵ��ĶԱȶȣ�D + 0.5 * dD^T * X
	������
	dog_pyr����˹��ֽ�����
	octv�����ص����ڵ���
	intvl�����ص����ڵĲ�
	r�����ص����ڵ���
	c�����ص����ڵ���
	xi���㷽���ϵ�����������
	xr��y�����ϵ�����������
	xc��x�����ϵ�����������
	����ֵ����ֵ��ĶԱȶ�
*/
static double interp_contr(IplImage*** dog_pyr, int octv, int intvl, 
							int r, int c, double xi, double xr, double xc)
{
	CvMat* dD, X, T;
	double t[1], x[3] = {xc, xr, xi};

	//ƫ�������������X
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	//����˷����T
	cvInitMatHeader(&T, 1, 1, CV_64FC1, t, CV_AUTOSTEP);

	dD = deriv_3D(dog_pyr, octv, intvl, r, c);
	//����˷���T = dD^T * X
	cvGEMM(dD, &X, 1, NULL, 0, &T, CV_GEMM_A_T);
	cvReleaseMat(&dD);

	return pixval32f(dog_pyr[octv][intvl], r, c) + t[0] * 0.5;
}

/*Ϊһ��feature�ṹ����ռ䲢��ʼ��
����ֵ����ʼ����ɵ�feature�ṹ��ָ��
*/
static struct feature* new_feature(void)
{
	struct feature* feat;	//������ָ��
	struct detection_data* ddata;	//�����������صĽṹ

	feat = malloc(sizeof(struct feature));
	memset(feat, 0, sizeof(struct feature));	//����
	ddata = malloc(sizeof(struct detection_data));
	memset(ddata, 0, sizeof(struct detection_data));

	feat->feature_data = ddata;
	feat->type = FEATURE_LOWE;

	return feat;
}