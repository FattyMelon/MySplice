#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include <string>
#include <iostream>
using namespace std;

extern "C"
{
#include "imgfeatures.h"
#include "sift.h"
#include "utils.h"
}

int main ()
{
	sift_features(NULL,NULL);
}