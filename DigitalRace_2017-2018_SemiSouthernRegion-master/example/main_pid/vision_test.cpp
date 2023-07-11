#include "api_kinect_cv.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "openni2.h"
#include "../openni2/Singleton.h"
#include <unistd.h>
#include "../sign_detection/SignDetection.h"
#include <chrono>
#include "signsRecognizer.h"
#include "extractInfo.h"
#include <stdlib.h>
#include "multilane.h"
#include "Hal.h"
#include "LCDI2C.h"
#include "api_i2c_pwm.h"
using namespace openni;
using namespace framework;
using namespace signDetection;
using namespace EmbeddedFramework;
#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#undef debug
#define debug false
#define SW1_PIN	160
#define SW2_PIN	161
#define SW3_PIN	163
#define SW4_PIN	164
#define SENSOR	165
#define LED		166
cv::Mat remOutlier(const cv::Mat &gray) {
    int esize = 1;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
    cv::Size( 2*esize + 1, 2*esize+1 ),
    cv::Point( esize, esize ) );
    cv::erode(gray, gray, element);
    std::vector< std::vector<cv::Point> > contours, polygons;
    std::vector< cv::Vec4i > hierarchy;
    cv::findContours(gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point> p;
        cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);
        polygons.push_back(p);
    }
    cv::Mat poly = cv::Mat::zeros(gray.size(), CV_8UC3);
    for (size_t i = 0; i < polygons.size(); ++i) {
        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::drawContours(poly, polygons, i, color, CV_FILLED);
    }
    return poly;
}
double getTheta(Point car, Point dst) {
    if (dst.x == car.x) return 0;
    if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}
///////// utilitie functions  ///////////////////////////

int main( int argc, char* argv[] ) {
	////// init videostream ///
	GPIO *gpio = new GPIO();
	I2C *i2c_device = new I2C();
	LCDI2C *lcd = new LCDI2C();
    	int sw1_stat = 1;
	int sw2_stat = 1;
	int sw3_stat = 1;
	int sw4_stat = 1;
	int sensor = 1;
	
	// Setup input
	gpio->gpioExport(SW1_PIN);
	gpio->gpioExport(SW2_PIN);
	gpio->gpioExport(SW3_PIN);
	gpio->gpioExport(SW4_PIN);
	gpio->gpioExport(SENSOR);
	gpio->gpioExport(LED);
	gpio->gpioSetDirection(SW1_PIN, INPUT);
	gpio->gpioSetDirection(SW2_PIN, INPUT);
	gpio->gpioSetDirection(SW3_PIN, INPUT);
	gpio->gpioSetDirection(SW4_PIN, INPUT);
	gpio->gpioSetDirection(SENSOR, INPUT);
	gpio->gpioSetDirection(LED, OUTPUT);
	i2c_device->m_i2c_bus = 2;
	
	if (!i2c_device->HALOpen()) {
		printf("Cannot open I2C peripheral\n");
		exit(-1);
	} else printf("I2C peripheral is opened\n");
	
	unsigned char data;
	if (!i2c_device->HALRead(0x38, 0xFF, 0, &data, "")) {
		printf("LCD is not found!\n");
		exit(-1);
	} else printf ("LCD is connected\n");
	usleep(10000);
	lcd->LCDInit(i2c_device, 0x38, 20, 4);
	lcd->LCDBacklightOn();
	lcd->LCDCursorOn();
	
	lcd->LCDSetCursor(3,1);
	lcd->LCDPrintStr("DRIVERLESS CAR");
	lcd->LCDSetCursor(5,2);
	lcd->LCDPrintStr("2017-2018");
	int dir = 0, throttle_val = 0;
	double theta = 0;
	int current_state = 0;
	char key = 0;

	    //=========== Init  =======================================================
	    ////////  Init PCA9685 driver   ///////////////////////////////////////////
	PCA9685 *pca9685 = new PCA9685() ;
	api_pwm_pca9685_init( pca9685 );
	if (pca9685->error >= 0)api_set_FORWARD_control( pca9685,throttle_val);
	    /// Init MSAC vanishing point library
	MSAC msac;
	api_vanishing_point_init( msac );
	int set_throttle_val = 0;
    throttle_val = 0;
   	theta = 0;
    if(argc == 2 ) set_throttle_val = atoi(argv[1]);
    fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
    int frame_width = VIDEO_FRAME_WIDTH;
    int frame_height = VIDEO_FRAME_HEIGHT;
    Point carPosition(frame_width / 2, frame_height);
    Point prvPosition = carPosition;
	bool running = false, started = false, stopped = false;
	OpenNI2::Instance() -> init();
	// signsRecognizer recognizer = signsRecognizer("/home/ubuntu/data/new_templates/templates.txt");
	ushort l_th = 600, h_th = 2000;
	vector<vector<Point> > regs;
	Mat depthImg, colorImg, grayImage, disparity;
    	while ( true )
    	{	
		Point center_point(0,0);
		key = getkey();
       	unsigned int bt_status = 0;
		unsigned int sensor_status = 0;
		gpio->gpioGetValue(SW4_PIN, &bt_status);
		gpio->gpioGetValue(SENSOR, &sensor_status);
		//std::cout<<sensor_status<<std::endl;
		if (!bt_status) {
			if (bt_status != sw4_stat) {
				running = !running;
				sw4_stat = bt_status;
				throttle_val = set_throttle_val;
			}
		} else sw4_stat = bt_status;
	
        if( key == 's') {
			running = !running;
			throttle_val = set_throttle_val;
			
		}
       	if( key == 'f') {
			fprintf(stderr, "End process.\n");
        	theta = 0;
        	throttle_val = 0;
	    	api_set_FORWARD_control( pca9685,throttle_val);
        	break;
		}
		if( !running )
		{
		lcd->LCDClear();
		lcd->LCDSetCursor(3,1);
		lcd->LCDPrintStr("PAUSE");
		continue;
		}
		if( running ){
		lcd->LCDClear();
		lcd->LCDSetCursor(3,1);
		lcd->LCDPrintStr("RUNNING");
		
		if (!sensor_status) {
			if (sensor_status != sensor) {
				running = !running;
				sensor = sensor_status;
				throttle_val = 0;
			}
		} else sensor = sensor_status;
			if (pca9685->error < 0)
           	 {
                cout<< endl<< "Error: PWM driver"<< endl<< flush;
                break;
           	 }
			if (!started)
			{
    			fprintf(stderr, "ON\n");
			    started = true; stopped = false;
				throttle_val = set_throttle_val;
                api_set_FORWARD_control( pca9685,throttle_val);
			}
        	auto st = chrono::high_resolution_clock::now();
			OpenNI2::Instance()->getData(colorImg, depthImg, grayImage, disparity);
			Mat colorTemp = colorImg.clone();
			auto bt = chrono::high_resolution_clock::now();
			vector<Rect> boxes;
			vector<int> labels;
			cv::Mat pyrDown;
			//SignDetection::Instance()->objectLabeling(boxes, labels, depthImg, colorImg, l_th, h_th, 1000, 8000, 50, 200, 1.5);
			cv::pyrDown( colorTemp, pyrDown, cv::Size(colorTemp.cols/2, colorTemp.rows/2));
			cv::Rect roi1 = cv::Rect(0, 240*3/4, 320, 240/4);
			cvtColor(pyrDown, grayImage, CV_BGR2GRAY);
            cv::Mat dst = keepLanes(grayImage, false);
            cv::imshow("dst", dst);
            cv::Point shift (0, 3 * grayImage.rows / 4);
        	bool isRight = true;
		//api_get_vanishing_point( grayImage, roi1, msac, center_point, true,"Wavelet");
            cv::Mat two = twoRightMostLanes(grayImage.size(), dst, shift, isRight);
           // cv::imshow("two", two);
			Rect roi2(0,   3*two.rows / 4, two.cols, two.rows / 4); //c?t ?nh
		
			Mat imgROI2 = two(roi2);
			cv::imshow("roi", imgROI2);
			int widthSrc = imgROI2.cols;
			int heightSrc = imgROI2.rows;
			vector<Point> pointList;
			//for (int y = 0; y < heightSrc; y++)
			//{
				for (int x = widthSrc; x >= 0; x--)
				{
					if (imgROI2.at<uchar>(30, x) == 255 )/////////////////25
					{
						pointList.push_back(Point(30, x));
						//break;
					}
					if(pointList.size() == 0){
						pointList.push_back(Point(30, 300));
					}
				
				}
			//}
			//std::cout<<"size"<<pointList.size()<<std::endl;
			int x = 0, y = 0;
			int xTam = 0, yTam = 0;
			for (int i = 0; i < pointList.size(); i++)
				{
					x = x + pointList.at(i).y;
					y = y + pointList.at(i).x;
				}
			xTam = (x / pointList.size());
			yTam = (y / pointList.size());
			xTam = xTam ;
			if(pointList.size()<=15&&pointList.size()>1)xTam = xTam - 70;
			yTam = yTam + 240 * 3 / 4;
			circle(grayImage, Point(xTam, yTam), 2, Scalar(255, 255, 0), 3);
			// imshow("result", grayImage);
            //if (center_point.x == 0 && center_point.y == 0) center_point = prvPosition;
            //prvPosition = center_point;
	    center_point = Point(xTam, yTam);
            double angDiff = getTheta(carPosition, center_point);
			//if(-20<angDiff&&angDiff<20)angDiff=0;
            theta = (angDiff*2);
			std::cout<<"angdiff"<<angDiff<<std::endl;
		       // theta = (0.00);
		    api_set_STEERING_control(pca9685,theta);
            int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
			auto et = chrono::high_resolution_clock::now();
		
			vector<int> signLabels;
			vector<string> names;
			// recognizer.labeling(boxes, labels, colorImg, signLabels, names);
			bt = chrono::high_resolution_clock::now();
		
		for (int i = 0; i < 0;i++) //names.size(); i++)
			{
				rectangle(colorImg, boxes[i], Scalar(255, 0, 0), 1, 8, 0);
				if (names[i] == "stop")
				{
					cout<<"dungxe";
					throttle_val = 0;
					theta = (0.00);	
					api_set_STEERING_control(pca9685,theta);
					api_set_FORWARD_control( pca9685,throttle_val);
					running = !running;
				}
				/*if (names[i] == "leftTurn")
				{
					cout<<"re trai";
					theta = (-60.00);	
					api_set_STEERING_control(pca9685,theta);
					api_set_FORWARD_control( pca9685,throttle_val);
					usleep(700000);//running = !running;
				}*/
				putText(colorImg, names[i], Point(boxes[i].x, boxes[i].y - 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
				
			}
		//cout<<"theta"<<theta<<endl;
		if (debug) printf("Sign_detection run in %.2fms\n", chrono::duration<double, milli> (et-bt).count());
			// imshow("color", colorImg);
			
			char ch = waitKey(10);
			if (ch == 'q')
				break;
			//////// End Detect traffic signs //////////////
	}
	}
    	return 0;
}


