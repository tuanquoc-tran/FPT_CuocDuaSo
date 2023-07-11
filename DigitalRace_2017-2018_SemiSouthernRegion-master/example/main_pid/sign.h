#ifndef __SIGN_H__
#define __SIGN_H__

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/ml.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
using namespace cv::ml;

class Sign
{
    private:
        HOGDescriptor _hog;
        Ptr<SVM> _svm;
        Rect _signRoi;
        int _class_id;

    public:
        Sign();
        void detect(Mat &mask);    
        void recognize(Mat &gray);
        void classify(Mat &gray_sign);
        int get_class_id();
};

// WARNING: should only be used once because of ycrcb equalization
void get_mask(Mat &img, Mat &mask, string colors)
{
    ycrcb_equalize(img);
    
    mask = Mat::zeros(img.cols, img.rows, CV_8UC1);
    if (colors.find("blue") != std::string::npos)
    {
        inRange(hsv, LOW_HSV_BLUE, HIG_HSV_BLUE, tmp_mask);
        mask = cv::bitwise_or(mask, tmp_mask, mask);
    }
    if (colors.find("red") != std::string::npos)
    {
        inRange(hsv, LOW_HSV_RED1, HIG_HSV_RED1, tmp_mask1);
        inRange(hsv, LOW_HSV_RED2, HIG_HSV_RED2, tmp_mask2);
        mask = cv::bitwise_or(tmp_mask1, tmp_mask2, mask);
    }
    if (colors.find("green") != std::string::npos)
    {
        inRange(hsv, LOW_HSV_GREEN, HIG_HSV_GREEN, tmp_mask);
        mask = cv::bitwise_or(mask, tmp_mask, mask);
    }

    Mat kernel = Mat::ones(KERNEL_SIZE, KERNEL_SIZE, CV_8UC1);

    dilate(mask, mask, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
}

void ycrcb_equalize(Mat &img)
{
	Mat ycrcb;
	cvtColor(img, ycrcb, CV_BGR2YCrCb);

	vector<Mat> chanels(3);
	split(ycrcb, chanels);

	Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
	clahe->apply(chanels[0], chanels[0]);

	merge(chanels, ycrcb);
	cvtColor(ycrcb, img, CV_YCrCb2BGR);
}
