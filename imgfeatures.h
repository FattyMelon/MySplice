/*
  此文件定义存储特征点的结构体feature，以及几个函数原型的声明：
1、特征点的导入和到处
2、特征点绘制
*/

#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "opencv\cxcore.h"

/*特征点类型：
FEATURE_OXFD表示牛津大学VGG提供的源码中的特征点格式
FEATURE_LOWE表示David.Lowe提供的源码中的特征点格式
*/
enum feature_type {
	FEATURE_OXFD,
	FEATURE_LOWE
};

/*特征点匹配类型：
FEATURE_FWD_MATCH：表明feature结构中的fwd_match域是对应的匹配点
FEATURE_BCK_MATCH：表明feature结构中的bck_match域是对应的匹配点
FEATURE_MDL_MATCH：表明feature结构中的mdl_match域是对应的匹配点
*/
enum feature_match_type
{
	FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH,
	FEATURE_MDL_MATCH
};

/*画出的特征点的颜色*/
#define FEATURE_OXFD_COLOR CV_RGB(255,255,0)
#define FEATURE_LOWE_COLOR CV_RGB(255,0,255)

/*最大特征描述子长度，定为128*/
#define FEATURE_MAX_D 128

/*特征点结构体
此结构体可存储2中类型的特征点：
FEATURE_OXFD表示是牛津大学VGG提供的源码中的特征点格式，
FEATURE_LOWE表示是David.Lowe提供的源码中的特征点格式。
如果是OXFD类型的特征点，结构体中的a,b,c成员描述了特征点周围的仿射区域(椭圆的参数)，即邻域。
如果是LOWE类型的特征点，结构体中的scl和ori成员描述了特征点的大小和方向。
fwd_match，bck_match，mdl_match一般同时只有一个起作用，用来指明此特征点对应的匹配点
*/
struct feature
{
	double x;						//特征点x的坐标
	double y;						//特征点y的坐标
	double a;
	double b;
	double c;
	double scl;						//LOWE特征点的尺度
	double ori;						//LOWE特征点的方向
	int d;							//特征描述子的维数，一般为128
	double descr[FEATURE_MAX_D];	//128维的特征描述子
	int type;						//特征点的类型，OXFD或LOWE
	int category;					//特征点的种类
	struct feature* fwd_match;		//
	struct feature* bck_match;
	struct feature* mdl_match;
	CvPoint2D64f img_pt;			//特征点的坐标
	CvPoint2D64f mdl_pt;			//匹配类型为mdl_match时点的坐标
	void* feature_data;				//用户定义的数据:
									//在SIFT极值点检测中，是detection_data结构的指针
									//在k-d树搜索中，是bbf_data结构的指针
									//在RANSAC算法中，是ransac_data结构的指针
};

/*从文件中读入图像特征
文件中的特征点格式必须是FEATURE_OXFD或FEATURE_LOWE格式
参数：
filename：文件名
type：特征点类型
feat：用来存储特征点的feature数组的指针,二维数组
返回值：导入的特征点个数
*/
extern int import_features(char* filename, int type, struct feature** feature);


#endif