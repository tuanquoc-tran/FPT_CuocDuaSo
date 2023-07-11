Sign::Sign()
{
    _hog = HOGDescriptor(Size(SIGN_SIZE, SIGN_SIZE),
					Size(SIGN_SIZE / 2, SIGN_SIZE / 2),
					Size(SIGN_SIZE / 4, SIGN_SIZE / 4),
					Size(SIGN_SIZE / 2, SIGN_SIZE / 2),
					9,
					1,
					-1,
					0,
					0.2,
					1,
					64,
					true);

	_svm = SVM::load("svm_model.xml");
    _sign_roi = Rect(0, 0, 0, 0);
    _class_id = 0;
}

void Sign::detect(Mat &mask)
{
	vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

    // set default is no sign found
    float max_area = 0;
    _sign_roi = Rect(0, 0, 0, 0);
    
    for (int i = 0; i < contours.size(); i++)
    {
        Rect bound = boundingRect(contours[i]);
        double contour_area = contourArea(contours[i]);
        if (contour_area <= max_area)
            continue;

        // constraints
        double ellipse_area = (3.14f * (double)(bound.width / 2) * (double)(bound.height / 2));
        if (contour_area >= MIN_SIGN_AREA)
            if ((1 - DIF_RATIO_SIGN_WIDTH_PER_HEIGHT < (float)bound.width / bound.height) && ((float)bound.width / bound.height < 1 + DIF_RATIO_SIGN_WIDTH_PER_HEIGHT))
                if ((1 - DIF_RATIO_SIGN_AREA < ((double)contour_area / ellipse_area)) && ((double)contour_area / ellipse_area < 1 + DIF_RATIO_SIGN_AREA))
                {
                    // update max sign
                    _sign_roi = bound;
                    max_area = contour_area;
                }
    }
}

void Sign::recognize(Mat &gray)
{
    // no sign
	if (_sign_roi.x == 0 && _sign_roi.y == 0 && _sign_roi.width == 0 && _sign_roi.height == 0)
		return NO_SIGN;
	
    // crop
	Mat sign_gray = gray(_sign_roi);
	resize(sign_gray, sign_gray, cv::Size(SIGN_SIZE, SIGN_SIZE));
	
	classify(sign_gray);
}

void Sign::classify(Mat &sign_gray)
{
    // compute HOG descriptor
	vector<float> descriptors;
	vector<Point> locations;
	_hog.compute(sign_gray, descriptors);
	Mat fm(descriptors, CV_32F);
	
    // predict matrix transposition
    _class_id = (int)(_svm->predict(fm.t()));
    if (_class_id != SIGN_LEFT && _class_id != SIGN_RIGHT && _class_id != SIGN_STOP)
        _class_id = NO_SIGN;
}

int get_class_id()
{
    return _class_id;
}