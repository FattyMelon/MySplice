/*
  此文件最重要
  包含SIFT特征点检测的实现
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
static void calc_feature_scales( CvSeq*, double, int );
static void adjust_for_img_dbl(CvSeq*);


static void calc_feature_oris( CvSeq*, IplImage*** );
static double* ori_hist( IplImage*, int, int, int, int, double );
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
static void smooth_ori_hist( double*, int );
static double dominant_ori( double*, int );
static void add_good_ori_features( CvSeq*, double*, int, double, struct feature* );
static struct feature* clone_feature( struct feature* );

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

	/*if (!feat)
	{
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	}*/

	//步骤一：建立尺度空间，即建立高斯查分（DoG）金字塔dog_pyr
	//转换为32位灰度图并归一化，然后进行一次高斯平滑
	init_img = create_init_img(img, img_dbl, sigma);
	//计算高斯金字塔的组数octvs
	octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;
	//在每一层的顶层用高斯模糊生成3幅图像，所以高斯金字塔每组有intvls+3层，DOG金字塔每组有intvls+2层
	//建立高斯金字塔gauss_pyr,是一个octvs*（intvls+3）的图像数组
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);

	//建立高斯差分（DoG）金字塔dog_pyr
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);

	//步骤二：在尺度空间中检测极值点，并进行精确定位和筛选
	//创建默认大小的内存存储器
	storage = cvCreateMemStorage(0);
	//在尺度空间中检测极值点
	features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, storage);
	//计算特征点序列features中每个特征点的尺度
	calc_feature_scales(features, sigma, intvls);
	//若设置了图像放大，则调整特征点坐标
	if(img_dbl)
		adjust_for_img_dbl(features);

	//步骤三：特征点方向赋值
	//计算每个特征点的梯度直方图，找出其主方向，若一个特征点不止一个主方向，将其分为2个特征点
	calc_feature_oris(features, gauss_pyr);


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
	gray = convert_to_gray32(img);

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

	//将8位单通道图像转换为32位单通道，并归一化（除以255）
	cvConvertScale(gray8, gray32, 1.0 / 255.0, 0);

	cvReleaseImage(&gray8);

	return gray32; //返回32位单通道图像
}

/*根据输入参数建立高斯金字塔
参数：
base：输入图像，作为高斯金字塔的基图像
octvs：高斯金字塔的组数
intvls：每组的层数
sigma：初始尺度
返回值：高斯金字塔，是一个octvs*(intvls+3)的图像数组
*/
static IplImage*** build_gauss_pyr(IplImage* base, int octvs, int intvls, double sigma)
{
	IplImage*** gauss_pyr;
	
	double* sig = calloc(intvls + 3, sizeof(double)); //每层的sigma参数数组
	double sig_total, sig_prev, k;
	int i, o;

	//为高斯金字塔gauss_pyr分配空间，共octvs个元素，每个元素是一组图像的首指针
	gauss_pyr = calloc(octvs, sizeof(IplImage**));
	//为第i组图像分配空间，共intvls+3个元素
	for (i = 0; i< octvs; ++i) 
		gauss_pyr[i] = calloc(intvls + 3, sizeof(IplImage* ));

	/*	计算sigma的公式
		sigma_{total}^2 = sigma_{i}^2 + sigma_{i-1}^2   */
	sig[0] = sigma;	//初试尺度
	k = pow(2.0, 1.0 / intvls);

	for (i = 1; i < intvls +3; ++i) 
	{
		sig_prev = pow(k, i - 1) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	//每组每层生成高斯金字塔
	for (o = 0; o < octvs; ++o) 
		for( i = 0; i < intvls + 3; ++i) 
		{
			if (o == 0 && i == 0)	//第0组，第0层，就是原图像
				gauss_pyr[o][i] = cvCloneImage(base);
			else if (i == 0)	//新的一组的首层图像是由上一组最后一层图像向下采样得到的
				gauss_pyr[o][i] = downsample(gauss_pyr[o-1][intvls]);
			else	//对上层图像进行高斯平滑得到当前层图像
			{
				//创建图像
				gauss_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i-1]), IPL_DEPTH_32F, 1);
				//对上一层图像gauss_pyr[o][i-1]进行参数为sig[i]的高斯平滑
				cvSmooth(gauss_pyr[o][i-1], gauss_pyr[o][i], CV_GAUSSIAN, 0, 0, sig[i], sig[i]);

			}
		}

	free(sig);	//释放sigma参数数组

	return gauss_pyr;
}

/*对输入图像做下采样生成其四分之一大小的图像(每个维度上减半)，使用最近邻差值方法
参数：
img：输入图像
返回值：下采样后的图像
*/
static IplImage* downsample(IplImage* img)
{
	//下采样图像
	IplImage* smaller = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);
	cvResize(img, smaller, CV_INTER_NN);	//最近邻插值
	
	return smaller;
}

/*通过对高斯金字塔中每相邻两层图像相减来建立高斯差分金字塔
参数：
gauss_pyr：高斯金字塔
octvs：组数
intvls：每组的层数
返回值：高斯差分金字塔，是一个octvs*(intvls+2)的图像数组
*/
static IplImage*** build_dog_pyr(IplImage*** gauss_pyr, int octvs, int intvls)
{
	IplImage*** dog_pyr;
	int i, o;

	//为高斯差分金字塔分配空间，共octvs个元素，每个元素是一组图像的首指针
	dog_pyr = calloc(octvs, sizeof(IplImage**));
	//为第i组图像dog_pyr[i]分配空间，共intvls+2个元素
	for (i = 0; i < octvs; ++i)
	{
		dog_pyr[i] = calloc(intvls + 2, sizeof(IplImage*));
	}
	
	//每组每层计算差分图像
	for (o = 0; o < octvs; ++o)
		for (i = 0; i < intvls + 2; ++i)
		{
			//创建DoG金字塔的第o组第i层的差分图像
			dog_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i]), IPL_DEPTH_32F, 1);
			//将高斯金字塔的第o组第i+1层图像减去第i层图像
			cvSub(gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL);
		}

	return dog_pyr;	//返回高斯差分金字塔
}

/*在尺度空间中检测极值点，通过插值精确定位，去除低对比度的点，去除边缘点，返回检测到的特征点序列
参数：
dog_pyr：高斯差分金字塔
octvs：高斯差分金字塔的组数
intvls：每组的层数
contr_thr：对比度阈值，针对归一化后的图像，用来去除不稳定特征
cur_thr：主曲率比值的阈值，用来去除边缘特征
storage：存储器
返回值：返回检测到的特征点的序列
*/
static CvSeq* scale_space_extrema(IplImage*** dog_pyr, int octvs, int intvls, 
									double contr_thr, int curv_thr, CvMemStorage* storage)
{
	CvSeq* features;	//特征点序列
	double prelim_contr_thr = 0.5 * contr_thr / intvls;	//像素对比度的阈值
	struct feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;

	//在存储器storage上创建存储极值点的序列，其中存储feature结构类型的数据
	features = cvCreateSeq(0, sizeof(CvSeq), sizeof(struct feature), storage);

	//遍历高斯差分金字塔，检测极值点
	//忽略边界线，只检测边界线以内的极值点
	for (o = 0; o < octvs; ++o)
		for (i = 1; i <= intvls; ++i)
			for (r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; ++r)	//第r行
				for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; ++c)	//第c列
					//进行初步的对比度检查，只有当归一化后的像素值大于对比度阈值prelim_contr_thr时才继续检测此像素点是否可能是极值
					if (ABS(pixval32f(dog_pyr[o][i], r, c)) > prelim_contr_thr)
					{
						//通过在尺度空间中将一个像素点的值与其周围3*3*3邻域内的点比较来决定此点是否极值点(极大值或极小都行)
						if (is_extremum(dog_pyr, o, i, r, c))	//若是极值点
						{
							//由于极值点的检测是在离散空间中进行的，所以检测到的极值点并不一定是真正意义上的极值点
							//因为真正的极值点可能位于两个像素之间，而在离散空间中只能精确到坐标点精度上
							//通过亚像素级插值进行极值点精确定位(修正极值点坐标)，并去除低对比度的极值点，将修正后的特征点组成feature结构返回
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);

							if (feat)
							{
								ddata = feat_detection_data(feat);
								//去除边缘响应
								if(!is_too_edge_like(dog_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, curv_thr))
								{
									cvSeqPush(features, feat);
								}
								else
									free(ddata);
								free(feat);
							}
						}
					}
	return features;					
}

/*通过在尺度空间中将一个像素点的值与其周围3*3*3邻域内的点比较来决定此点是否极值点(极大值或极小都行)
参数：
dog_pyr：高斯差分金字塔
octv：像素点所在的组
intvl：像素点所在的层
r：像素点所在的行
c：像素点所在的列
返回值：若指定的像素点是极值点(最大值或者最小值)，返回1；否则返回0
*/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	float val = pixval32f(dog_pyr[octv][intvl], r, c);
	int i, j, k;

	//检查是否最大值
	if (val > 0)
	{
		for (i = -1; i <= 1; ++i)
			for (j = -1; j <= 1; ++j)
				for (k = -1; k <= 1; ++k)
					if (val < pixval32f(dog_pyr[octv][intvl+i], r + j, c + k))
						return 0;
	}
	//检查是否最小值
	else
	{
		for( i = -1; i <= 1; ++i)//层
			for( j = -1; j <= 1; ++j )//行
				for( k = -1; k <= 1; ++k )//列
					if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k))
						return 0;
	}
}

/*通过亚像素级插值进行极值点精确定位(修正极值点坐标)，并去除低对比度的极值点，将修正后的特征点组成feature结构返回
参数：
dog_pyr：高斯差分金字塔
octv：像素点所在的组
intvl：像素点所在的层
r：像素点所在的行
c：像素点所在的列
intvls：每组的层数
contr_thr：对比度阈值，针对归一化后的图像，用来去除不稳定特征
返回值：返回经插值修正后的特征点(feature类型)；若经有限次插值依然无法精确到理想情况或者该点对比度过低，返回NULL
*/
static struct feature* interp_extremum(IplImage*** dog_pyr, int otcv, int intvl, 
										int r, int c, int intvls, double contr_thr)
{
	struct feature* feat;	//修正后的特征点
	struct detection_data * ddata;	//与特征检测有关的结构，存在feature结构的feature_data成员中
	double xi, xr, xc, contr;
	int i = 0;	//插值次数

	//
	while (i < SIFT_MAX_INTERP_STEPS)
	{
		//进行一次极值点差值，计算σ(层方向,intvl方向)，y，x方向上的子像素偏移量(增量)
		interp_step(dog_pyr, otcv, intvl, r, c, &xi, &xr, &xc);
		//若在任意方向上的偏移量大于0.5，意味着差值中心已经偏移到它的临近点上，所以必须改变当前关键点的位置坐标
		if (ABS(xi) < 0.5 && ABS(xr) < 0.5 && ABS(xc) < 0.5)	//符合条件跳出循环
			break;

		//修正关键点的坐标
		c += cvRound(xc);
		r += cvRound(xr);
		intvl += cvRound(xi);

		//若坐标修正后超出范围，则结束插值，返回NULL
		if (intvl < 1 || intvl > intvls ||	//层坐标越界
			c < SIFT_IMG_BORDER ||	c >= dog_pyr[otcv][0]->width - SIFT_IMG_BORDER ||	//行坐标到边界线内
			r < SIFT_IMG_BORDER ||	r >= dog_pyr[otcv][0]->height - SIFT_IMG_BORDER)
		{
			return NULL;
		}

		++i;
	}

	if (i >= SIFT_MAX_INTERP_STEPS)
		return NULL;

	//计算被插值点的对比度:D + 0.5 * dD^T * X
	contr = interp_contr(dog_pyr, otcv, intvl, r, c, xi, xr, xc);
	if (ABS(contr) < contr_thr / intvls)	//若该点对比度太小，舍弃
		return NULL;

	//为一个特征点feature结构分配空间并初始化，返回特征点指针
	feat = new_feature();
	ddata = feat_detection_data(feat);

	//将修正后的坐标赋值给特征点feat
	feat->img_pt.x = feat->x = (c + xc) * pow(2.0, otcv);
	feat->img_pt.y = feat->y = (r + xr) * pow(2.0, otcv);

	ddata->r = r;
	ddata->c = c;
	ddata->octv = otcv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;	//特征点在层方向(σ方向,intvl方向)上的亚像素偏移量

	return feat;
}

/*进行一次极值点差值，计算x，y，σ方向(层方向)上的子像素偏移量(增量)
	参数：
	dog_pyr：高斯差分金字塔
	octv：像素点所在的组
	intvl：像素点所在的层
	r：像素点所在的行
	c：像素点所在的列
	xi：输出参数，层方向上的子像素增量(偏移)
	xr：输出参数，y方向上的子像素增量(偏移)
	xc：输出参数，x方向上的子像素增量(偏移)
*/
static void interp_step(IplImage*** dog_pyr, int octv, int intvl, 
						int r, int c, double* xi, double* xr, double* xc)
{
	CvMat* dD, * H, * H_inv, X;
	double x[3] = {0};

	//在DoG金字塔中计算某点的x方向、y方向以及尺度方向的偏导数
	dD = deriv_3D(dog_pyr, octv, intvl, r, c);

	//在DoG金字塔中计算某点的3*3海森矩阵
	H = hessian_3D(dog_pyr, octv, intvl, r, c);
	H_inv = cvCreateMat(3, 3, CV_64FC1);
	//求逆矩阵
	cvInvert(H, H_inv, CV_SVD);
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	//X = - H^(-1) * dD，H的三个元素分别是x,y,σ方向上的偏移量
	cvGEMM(H_inv, dD, -1, NULL, 0, &X, 0);

	cvReleaseMat(&dD);	
	cvReleaseMat(&H);
	cvReleaseMat(&H_inv);

	*xi = x[2];	//层方向的偏移量
	*xr = x[1];
	*xc = x[0];
}

/*在DoG金字塔中计算某点的x方向、y方向以及尺度方向上的偏导数
参数：
dog_pyr：高斯差分金字塔
octv：像素点所在的组
intvl：像素点所在的层
r：像素点所在的行
c：像素点所在的列
返回值：返回3个偏导数组成的列向量{ dI/dx, dI/dy, dI/ds }^T
*/
static CvMat* deriv_3D(IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	CvMat* dI;
	double dx, dy, ds;

	//求差分来代替偏导,这里是用的隔行求差取中值的梯度计算方法
	//求x方向上的差分来近似代替偏导数
	dx = (pixval32f(dog_pyr[octv][intvl], r, c+1) -
		pixval32f(dog_pyr[octv][intvl], r, c-1)) / 2.0;
	//求y方向上的差分来近似代替偏导数
	dy = ( pixval32f(dog_pyr[octv][intvl], r+1, c) -
		pixval32f(dog_pyr[octv][intvl], r-1, c)) / 2.0;
	//求层间的差分来近似代替尺度方向上的偏导数
	ds = ( pixval32f(dog_pyr[octv][intvl+1], r, c) -
		pixval32f(dog_pyr[octv][intvl-1], r, c )) / 2.0;

	//组成列向量
	dI = cvCreateMat(3, 1, CV_64FC1);
	cvmSet(dI, 0, 0, dx);
	cvmSet(dI, 1, 0, dy);
	cvmSet(dI, 2, 0, ds);

	return dI;
}

/*在DoG金字塔中计算某点的3*3海森矩阵
    / Ixx  Ixy  Ixs \
    | Ixy  Iyy  Iys |
    \ Ixs  Iys  Iss /
参数：
dog_pyr：高斯差分金字塔
octv：像素点所在的组
intvl：像素点所在的层
r：像素点所在的行
c：像素点所在的列
返回值：返回3*3的海森矩阵
*/
static CvMat* hessian_3D(IplImage*** dog_pyr, int octv, int intvl,
							int r, int c)
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f(dog_pyr[octv][intvl], r, c);	//该点的像素值

	//用差分近似代替导数
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

	//组成海森矩阵
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

/*计算被插值点的对比度：D + 0.5 * dD^T * X
	参数：
	dog_pyr：高斯差分金字塔
	octv：像素点所在的组
	intvl：像素点所在的层
	r：像素点所在的行
	c：像素点所在的列
	xi：层方向上的子像素增量
	xr：y方向上的子像素增量
	xc：x方向上的子像素增量
	返回值：插值点的对比度
*/
static double interp_contr(IplImage*** dog_pyr, int octv, int intvl, 
							int r, int c, double xi, double xr, double xc)
{
	CvMat* dD, X, T;
	double t[1], x[3] = {xc, xr, xi};

	//偏移量组成列向量X
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	//矩阵乘法结果T
	cvInitMatHeader(&T, 1, 1, CV_64FC1, t, CV_AUTOSTEP);

	dD = deriv_3D(dog_pyr, octv, intvl, r, c);
	//矩阵乘法：T = dD^T * X
	cvGEMM(dD, &X, 1, NULL, 0, &T, CV_GEMM_A_T);
	cvReleaseMat(&dD);

	return pixval32f(dog_pyr[octv][intvl], r, c) + t[0] * 0.5;
}

/*为一个feature结构分配空间并初始化
返回值：初始化完成的feature结构的指针
*/
static struct feature* new_feature(void)
{
	struct feature* feat;	//特征点指针
	struct detection_data* ddata;	//与特征检测相关的结构

	feat = malloc(sizeof(struct feature));
	memset(feat, 0, sizeof(struct feature));	//清零
	ddata = malloc(sizeof(struct detection_data));
	memset(ddata, 0, sizeof(struct detection_data));

	feat->feature_data = ddata;
	feat->type = FEATURE_LOWE;

	return feat;
}

/*去除边缘响应，即通过计算主曲率比值判断某点是否边缘点
参数：
dog_img：此特征点所在的DoG图像
r：特征点所在的行
c：特征点所在的列
cur_thr：主曲率比值的阈值，用来去除边缘特征
返回值：0：此点是非边缘点；1：此点是边缘点
*/
static int is_too_edge_like(IplImage* dog_img, int r, int c, int curv_thr)
{
	double d, dxx, dyy, dxy, tr, det;

	d = pixval32f(dog_img, r, c);

	//用差分近似代替偏导，求出海森矩阵的几个元素值
    /*  / dxx  dxy \
        \ dxy  dyy /   */
	dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
	dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
	dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
		pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
	tr = dxx + dyy;//海森矩阵的迹
	det = dxx * dyy - dxy * dxy;//海森矩阵的行列式

	//若行列式为负，表示曲率有不同的符号，去除此点
	//1代表点是边缘点，0代表点不是边缘点
	if (det <= 0)
		return 1;
	//通过式子：(r+1)^2/r 判断主曲率的比值是否满足条件，若小于阈值，表明不是边缘点
	if (tr * tr / det < (curv_thr + 1.0)*(curv_thr + 1.0) / curv_thr)
		return 0;
	return 1;
}

/*计算特征点序列中每个特征点的尺度
参数：
features：特征点序列
sigma：初始高斯平滑参数，即初始尺度
intvls：尺度空间中每组的层数
*/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
	struct feature* feat;
	struct detection_data* ddata;
	double intvl;
	int i, n;

	n = features->total;	//总的特征点个数

	for (i = 0; i < n; ++i) 
	{
		feat = CV_GET_SEQ_ELEM(struct feature, features, i);
		ddata = feat_detection_data(feat);
		//根据公式计算特征点的尺度
		intvl = ddata->intvl + ddata->subintvl;
		feat->scl = sigma * pow(2.0, ddata->octv + intvl / intvls);
		ddata->scl_octv = sigma * pow(2.0, intvl / intvls);
	}
}

/*将特征点序列中每个特征点的坐标减半(当设置了将图像放大为原图的2倍时，特征点检测完之后调用)
参数：
features：特征点序列
*/
static void adjust_for_img_dbl(CvSeq* features)
{
	struct feature* feat;
	int i, n;

	n = features->total;

	for (i = 0; i < n; ++i)
	{
		feat = CV_GET_SEQ_ELEM(struct feature, features, i);
		//将特征点的x,y坐标和尺度都减半
		feat->x /= 2.0;
		feat->y /= 2.0;
		feat->scl /= 2.0;
		feat->img_pt.x /= 2.0;
		feat->img_pt.y /= 2.0;
	}
}

/*计算每个特征点的梯度直方图，找出其主方向，若一个特征点有不止一个主方向，将其分为两个特征点
参数：
features：特征点序列
gauss_pyr：高斯金字塔
*/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr)
{
	struct feature* feat;
	struct detection_data* ddata;
	double* hist;	//存放梯度直方图的数组
	double omax;
	int i, j, n = features->total;	//特征点的个数

	//遍历特征点序列
	for (i = 0; i < n; ++i)
	{
		feat = malloc(sizeof(struct feature));
		//移除列首元素，放到feat中
		cvSeqPopFront(features, feat);
		ddata = feat_detection_data(feat);

		//计算指定像素点的梯度方向直方图
		hist = ori_hist(gauss_pyr[ddata->octv][ddata->intvl],
						ddata->r, ddata->c,
						SIFT_ORI_HIST_BINS,
						cvRound(SIFT_ORI_RADIUS * ddata->scl_octv),
						SIFT_ORI_SIG_FCTR * ddata->scl_octv);

		//对梯度直方图进行高斯平滑
		for (j = 0; j < SIFT_ORI_SMOOTH_PASSES; ++j)
			smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);

		//查找梯度直方图中主方向的梯度幅值，即查找直方图中最大bin的值,返回给omax
		omax = dominant_ori(hist, SIFT_ORI_HIST_BINS);

		//当存在一个相当于主峰值能量80%能量的峰值时，则将这个方向认为是该特征点的辅方向
		add_good_ori_features(features, hist, SIFT_ORI_HIST_BINS,
								omax * SIFT_ORI_PEAK_RATIO, feat);

		//释放内存
		free(ddata);
		free(feat);
		free(hist);
	}
}

/*计算指定像素点的梯度方向直方图，返回存放直方图的数组
参数：
img：图像指针
r：特征点所在的行
c：特征点所在的列
n：直方图中柱(bin)的个数，默认是36
rad：区域半径，在此区域中计算梯度方向直方图
sigma：计算直翻图时梯度幅值的高斯权重的初始值
返回值：返回一个n元数组，其中是方向直方图的统计数据
*/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad, double sigma)
{
	double* hist;	//直方图数组
	double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	//为直方图数组分配空间，共n个元素，n是柱的个数
	hist = calloc(n, sizeof(double));
	exp_denom = 2.0 * sigma * sigma;

	//遍历以指定点为中心的搜索区域
	for(i = -rad; i <= rad; ++i)
		for(j = -rad; j <= rad; ++j)
			if(calc_grad_mag_ori(img, r+i, c+j, &mag, &ori))
			{
				w = exp(-(i*i + j*j) / exp_denom);	//	该点的梯度幅值权重
				bin = cvRound(n * (ori + CV_PI) / PI2);
				bin = (bin < n)? bin : 0;
				hist[bin] += w * mag;
			}

	return hist;
}

/*计算指定点的梯度的幅值magnitude和方向orientation
参数：
img：图像指针
r：特征点所在的行
c：特征点所在的列
mag：输出参数，此点的梯度幅值
ori：输出参数，此点的梯度方向
返回值：如果指定的点是合法点并已计算出幅值和方向，返回1；否则返回0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c,
								double* mag, double* ori)
{
	double dx, dy;

	//对输入的坐标值进行检查
	if (r > 0 && r < img->height - 1 && c > 0 && c < img->width - 1)
	{
		//用差分近似代替偏导，来求梯度的幅值和方向
		dx = pixval32f(img, r, c+1) - pixval32f(img, r, c-1);
		dy = pixval32f(img, r+1, c) - pixval32f(img, r-1, c);
		*mag = sqrt(dx*dx + dy*dy);
		*ori = atan2(dy, dx);
		return 1;
	}
	//行列坐标不合法，返回0
	else
		return 0;
}

/*对梯度方向直方图进行高斯平滑，弥补因没有仿射不变性而产生的特征点不稳定的问题
参数：
hist：存放梯度直方图的数组
n：梯度直方图中bin的个数
*/
static void smooth_ori_hist(double* hist, int n)
{
	double prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n-1];
	//类似均值漂移的一种领域平滑，减少突变的影响
	for(i = 0; i < n; ++i)
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * ((i+1 == n)? h0 : hist[i+1]);
		prev =tmp;
	}
}

/*查找梯度直方图中主方向的梯度幅值，即查找直方图中最大bin的值
参数：
hist：存放直方图的数组
n：直方图中bin的个数
返回值：返回直方图中最大的bin的值
*/
static double dominant_ori(double* hist, int n)
{
	double omax;
	int maxbin, i;

	omax = hist[0];
	maxbin = 0;

	//遍历直方图，找到最大的bin
	for (i = 1; i < n; ++i)
	{
		if (hist[i] > omax)
		{
			omax = hist[i];
			maxbin = i;
		}
	}
	return omax;
}

/*若当前特征点的直方图中某个bin的值大于给定的阈值，则新生成一个特征点并添加到特征点序列末尾
  传入的特征点指针feat是已经从特征点序列features中移除的，所以即使此特征点没有辅方向(第二个大于幅值阈值的方向)
  也会执行一次克隆feat，对其方向进行插值修正，并插入特征点序列的操作
参数：
features：特征点序列
hist：梯度直方图
n：直方图中bin的个数
mag_thr：幅值阈值，若直方图中有bin的值大于此阈值，则增加新特征点
feat：一个特征点指针，新的特征点克隆自feat，但方向不同
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n,
									double mag_thr, struct feature* feat)
{
	struct feature* new_feat;
	double bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	//遍历直方图
	for (i = 0; i < n; ++i)
	{
		l = (i == 0)? n-1 : i-1;	//前一个bin的下标
		r = (i + 1) % n;	//后一个bin的下标

		//若当前的bin是局部极值，并且值大于给定的幅值阈值，则新生成一个新的特征点并添加到特征点序列末尾
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			//根据左、中、右三个bin的值对当前bin进行直方图插值
			
		}
	}
}