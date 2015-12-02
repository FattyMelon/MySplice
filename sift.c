/*
  此文件最重要
  包含SIFT特征点检测的实现
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

/*使用默认参数在图像中提取SIFT特征点
参数：
img：图像指针
feat：用来存储特征点的feature数组的指针
      此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*feat)
返回值：提取的特征点个数，若返回-1表明提取失败
*/
int sift_features(IplImage* img, struct feature** feat)
{
	//调用_sift_features()函数进行特征点检测
	return _sift_features( img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
							SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
							SIFT_DESCR_HIST_BINS );
}

/*使用用户指定的参数在图像中提取SIFT特征点
参数：
img：输入图像
feat：存储特征点的数组的指针
      此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*feat)
intvls：每组的层数
sigma：初始高斯平滑参数σ
contr_thr：对比度阈值，针对归一化后的图像，用来去除不稳定特征
curv_thr：去除边缘的特征的主曲率阈值
img_dbl：是否将图像放大为之前的两倍
descr_width：特征描述过程中，计算方向直方图时，将特征点附近划分为descr_width*descr_width个区域，每个区域生成一个直方图
descr_hist_bins：特征描述过程中，每个直方图中bin的个数
返回值：提取的特征点个数，若返回-1表明提取失败
*/

int _sift_features( IplImage* img, struct feature** feat, int intvls,
					double sigma,double contr_thr, int curv_thr, int img_dbl, 
					int descr_width, int descr_hist_bins )
{
	IplImage* init_img;	//初始化后的图像
	IplImage*** gauss_pyr, *** dog_pyr;	//三级指针，高斯金字塔图像组，DoG金字塔图像组
	CvMemStorage* storage; //存储器
	CvSeq* features; //存储特征点的序列，序列中存放的是struct feature类型的指针
	int octvs, i, n = 0;

	//输入参数检查
	if (!img) 
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}

	if (!feat)
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}

	//#步骤一：建立尺度空间，即建立高斯查分（DoG）金字塔dog_pyr
	//转换为32位灰度图并归一化，然后进行一次高斯平滑
	init_img = create_init_img(img, img_dbl, sigma);

	return n;
}


/************************ Functions prototyped here **************************/

/*将原图转换为32位灰度图并归一化，然后进行一次高斯平滑，并根据参数img_dbl决定是否将图像尺寸放大为原图的2倍
参数：
img：输入的原图像
img_dbl：是否将图像放大为之前的两倍
sigma：初始高斯平滑参数σ
返回值：初始化完成的图像
*/
static IplImage* create_init_img(IplImage* img, int img_dbl, double sigma)
{
	IplImage* gray, * dbl;
	float sig_diff;

	//调用函数，将输入图像转换为32位灰度图，并归一化
	//gray = convert_to_gray32(img);

	//若设置了将图像放大为原图的2倍
	if (img_dbl)
	{
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
		dbl = cvCreateImage(cvSize(img->width * 2, img->height * 2), IPL_DEPTH_32F, 1);   //创建放大图像
		cvResize(gray, dbl, CV_INTER_CUBIC);  //放大原图的尺寸，三次样条插值

		//高斯平滑，高斯核在x，y方向上的标准差都是sig_diff
		cvSmooth(dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);
		cvReleaseImage(&gray);
		
		return dbl;
	}
	else
	{
		//计算第0层的尺度
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
		//高斯平滑
		cvSmooth(gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);

		return gray;
	}
}

/*将输入图像转换为32位灰度图,并进行归一化
参数：
img：输入图像，3通道8位彩色图(BGR)或8位灰度图
返回值：32位灰度图
*/
static IplImage* convert_to_gray32(IplImage* img)
{
	IplImage* gray8, * gray32;
	
	//创建32位单通道图像
	gray32 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	
	//首先将原图转换为8位单通道图像
	if (img->nChannels == 1) //若原图本身就是单通道，则直接复制
		gray8 = cvClone(img);
	else //若原图是三通道图像
	{
		gray8 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray8, CV_BGR2GRAY);  //将原图转换为8位单通道图像
	}

	//将8位当通道图像转换为32位单通道，并归一化（除以255）
	cvConvertScale(gray8, gray32, 1.0 / 255.0, 0);

	cvReleaseImage(&gray8);

	return gray32; //返回32位单通道图像
}