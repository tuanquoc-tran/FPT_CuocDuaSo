/**
    This code runs our car automatically and log video, controller (optional)
    Line detection method: Canny
    Targer point: vanishing point
    Control: pca9685
    
    You should understand this code run to image how we can make this car works from image processing coder's perspective.
    Image processing methods used are very simple, your task is optimize it.
    Besure you set throttle val to 0 before end process. If not, you should stop the car by hand.
    In our experience, if you accidental end the processing and didn't stop the car, you may catch it and switch off the controller physically or run the code again (press up direction button then enter).
**/
#include "api_kinect_cv.h"
// api_kinect_cv.h: manipulate openNI2, kinect, depthMap and object detection
#include "api_lane_detection.h"
// api_lane_detection.h: manipulate line detection, finding lane center and vanishing point
#include "api_i2c_pwm.h"
#include "multilane.h"
#include <iostream>
#include "Hal.h"
#include "LCDI2C.h"
using namespace openni;
using namespace EmbeddedFramework;
#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define VIDEO_FRAME_WIDTH 320	
#define VIDEO_FRAME_HEIGHT 240

<<<<<<< HEAD
#define IMG_DIR "img9.png"

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

#define LOW_HSV_BLUE Scalar(109, 60, 120)
#define HIG_HSV_BLUE Scalar(133, 255, 255)

#define KERNEL_SIZE 5
#define SIGN_SIZE 100
#define MIN_RATIO_BOUND_WIDTH_PER_FRAME_WIDTH 0.05
#define DIF_RATIO_BOUND_WIDTH_PER_HEIGHT 0.5
#define DIF_RATIO_CONTOUR_AREA 0.1
#define MIN_CONTOUR_AREA 1800

#define SW1_PIN 160
#define SW2_PIN 161
#define SW3_PIN 163
#define SW4_PIN 164
#define SENSOR 165
#define GREEN_MIN Scalar(34,138,12)
#define GREEN_MAX Scalar(83,246,124)
#define AREA_MIN 17000

enum SIGN_INDEX
{
    NO_SIGN = 0,
    SIGN_STOP = 1;
    SIGN_LEFT = 2,
    SIGN_RIGHT = 3,
};

bool flagTurn = false;
int dirTurn = 0;
Mat colorImg;
//bool getFrame = false;
bool endThread = true;
bool getFrame = false;

int preDetect = NO_SIGN;
int countDetect = 0;
int detectLeft = 0;
int detectRight = 0;
int set_throttle_val = 0;
int throttle_config = 0;
bool test_obt= false;
PCA9685 *pca9685 = new PCA9685();

cv::Mat remOutlier(const cv::Mat &gray)
{
=======
#define SW1_PIN	160
#define SW2_PIN	161
#define SW3_PIN	163
#define SW4_PIN	164
#define SENSOR	166
cv::Mat remOutlier(const cv::Mat &gray) {
>>>>>>> parent of 035dac7... Semi Round
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
<<<<<<< HEAD

char analyzeFrame(const VideoFrameRef &frame_depth, const VideoFrameRef &frame_color, Mat &depth_img, Mat &color_img)
{
    DepthPixel *depth_img_data;
    RGB888Pixel *color_img_data;
=======
char analyzeFrame(const VideoFrameRef& frame_depth,const VideoFrameRef& frame_color,Mat& depth_img, Mat& color_img) {
    DepthPixel* depth_img_data;
    RGB888Pixel* color_img_data;
>>>>>>> parent of 035dac7... Semi Round

    int w = frame_color.getWidth();
    int h = frame_color.getHeight();

    depth_img = Mat(h, w, CV_16U);
    color_img = Mat(h, w, CV_8UC3);
    Mat depth_img_8u;
	

            depth_img_data = (DepthPixel*)frame_depth.getData();

            memcpy(depth_img.data, depth_img_data, h*w*sizeof(DepthPixel));

            normalize(depth_img, depth_img_8u, 255, 0, NORM_MINMAX);

            depth_img_8u.convertTo(depth_img_8u, CV_8U);
            color_img_data = (RGB888Pixel*)frame_color.getData();

            memcpy(color_img.data, color_img_data, h*w*sizeof(RGB888Pixel));

            cvtColor(color_img, color_img, COLOR_RGB2BGR);
		
            return 'c';
}

/// Return angle between veritcal line containing car and destination point in degree
double getTheta(Point car, Point dst) {
    if (dst.x == car.x) return 0;
    if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

<<<<<<< HEAD
/////// My function
cv::Mat filterLane(const cv::Mat &imgLane, bool &pop, Point &point, int check, bool &preState)
{
    pop = false;
    point.x = 0;
    point.y = 0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgLane, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    if (contours.size() == 0)
    {
        cv::Mat none = cv::Mat::zeros(imgLane.size(), CV_8UC1);
        return none;
    }
    cv::Mat result = cv::Mat::zeros(imgLane.size(), CV_8UC1);
    int sumX = 0;
    int sumY = 0;
    int maxArea = 0;
    int maxIndex = 0;
    for (int i = 0; i < (int)contours.size(); ++i)
    {
        int s = cv::contourArea(contours[i]);
        if (s > maxArea)
        {
            maxArea = s;
            maxIndex = i;
        }
    }
    if (cv::contourArea(contours[maxIndex]) < 100)
    {
        cv::Mat none = cv::Mat::zeros(imgLane.size(), CV_8UC1);
        return none;
    }
    int xMin = 0, yMin = 1000, xMax = 0, yMax = -1;
    for (int i = 0; i < contours[maxIndex].size(); i++)
    {
        // if (contours[maxIndex][i].y > yMax)
        // {
        //     yMax = contours[maxIndex][i].y;
        //     xMax = contours[maxIndex][i].x;
        // }
        // if (contours[maxIndex][i].y < yMin)
        // {
        //     yMin = contours[maxIndex][i].y;
        //     xMin = contours[maxIndex][i].x;
        // }
        sumX += contours[maxIndex][i].x;
        sumY += contours[maxIndex][i].y;
    }

    //if((xMax>=xMin) && (check==1)){
    cv::drawContours(result, contours, maxIndex, cv::Scalar(255), CV_FILLED);
    point.x = sumX / contours[maxIndex].size();
    point.y = sumY / contours[maxIndex].size();

	if ((point.x  > imgLane.cols/2) && (check == -1) && (preState == false))
    {
	//cout << "sum X: " << sumX << endl;
        cv::Mat none = cv::Mat::zeros(imgLane.size(), CV_8UC1);
        return none;
    }
    if ((point.x  < imgLane.cols / 2) && (check == 1) && (preState == false))
    {
        cv::Mat none = cv::Mat::zeros(imgLane.size(), CV_8UC1);
        return none;
    }


    pop = true;
    return result;
    // } else
    // if((xMax<=xMin) && (check==-1)){
    //     cv::drawContours(result, contours, maxIndex, cv::Scalar(255), CV_FILLED);
    //     point.x = sumX / contours[maxIndex].size();
    //     point.y = sumY / contours[maxIndex].size();
    //     pop = true;
    //     return result;
    // }
    // return result;
}

///////////////////////cose test

void get_obtacle(Mat depthMap, int &xL, int &xR, int &y, int &w)
{
    ////
   /// imshow("depthMap",depthMap);
////
    using namespace std;
    using namespace cv;
     // Init//
    int slice_nb = 3;
    int lower_slice_idx = 3;
    int upper_slice_idx = lower_slice_idx + slice_nb;
    //int lower_bound = DIST_MIN + lower_slice_idx * SLICE_DEPTH;
    //int upper_bound = lower_bound + slice_nb*SLICE_DEPTH;
    int lower_bound = 5;//20; /////xa gan
    int upper_bound = 50;//50;
	test_obt= false;

    
    //Mat depthMap;
    Mat grayImage;
    vector< Rect > rects;
    Rect intersect;
    vector< Rect > output_boxes;

    Rect roi_1 = Rect(0, grayImage.rows/4,
                    grayImage.cols, grayImage.rows/2);
    Rect roi_2(0,32 , 320, 118);

    int frame_width = 320; //capture.get( CV_CAP_PROP_FRAME_WIDTH );
    int frame_height = 240; //capture.get( CV_CAP_PROP_FRAME_HEIGHT );

    //api_kinect_cv_get_images(capture, depthMap , grayImage); //get depth

    Mat crop_grayImage = grayImage(roi_1);
    Mat crop_depthMap = depthMap(roi_1);

    test_obt = api_kinect_cv_get_obtacle_rect( depthMap, output_boxes, roi_2, lower_bound, upper_bound );/////////--kiem tra co vat can hay khong 
	if (test_obt == true) cout<< "co vat can" << endl;
	else cout<<"khong vat can\n";
//--------------------------------------------------------------
	
    Mat binImg = Mat::zeros(depthMap.size(), CV_8UC1);

    api_kinect_cv_center_rect_gen( rects, frame_width, frame_height);
    Rect center_rect = rects[lower_slice_idx];
    center_rect = center_rect + Size(0, slice_nb*SLICE_DEPTH); // lay center_rect
	int temp=0;

    for( int i = 0; i< output_boxes.size(); i++ )
        {
//                    rectangle( binImg, output_boxes[i], Scalar( 255) );
            //intersect = output_boxes[i] & center_rect;
            intersect = output_boxes[i];
            if( intersect.area() != 0 && ((float)intersect.width/intersect.height<1.2 || (float)intersect.width/intersect.height>0.2))
               {
                    rectangle( binImg, intersect, Scalar( 255) );
                    //cout<< endl<< "Has Collision detect"<< flush;
			temp++;
			cout <<temp;
                }

		else cout <<"intersect = 0"<<endl;
        }
        //lay toa do, rong, cao.
    xL = intersect.x;
    xR = intersect.x + intersect.width;
    y = intersect.y;
    w = intersect.width;
    //int h = intersect.height;
    imshow( "BoundingRect", binImg );
    if(!grayImage.empty())
        imshow( "gray", crop_grayImage );
/*   if (!depthMap.empty() )
        imshow( "depth", crop_depthMap );
*/
}

void PointCenter_Displacement(int &xTam, int xL, int xR)
{
    int dodaixe = 30;
    int distan = 7;

    if (xTam < xL)
    {
        if (  xTam > (xL - dodaixe/2 - distan) )
            xTam = xL + dodaixe/2 + distan;
        else
            xTam = xTam;
    }
    else if (xTam > xR)
    {
        if (  xTam < (xR + dodaixe/2 + distan) )
            xTam = xL + dodaixe/2 + distan;
        else
            xTam = xTam;
    }
    else
    {
	cout << "AAAAAAAAAAAAAAA";
    }
}

void controlTurn(PCA9685 *&pca9685, int dir)
{
    if (dir == SIGN_LEFT)
    {
        double theta = ALPHA * 78;
        api_set_STEERING_control(pca9685, theta);
        cout << "Turn Left==================================================" << endl;
        sleep(1);
        cout << "Normal" << endl;
    }
    else if (dir == SIGN_RIGHT)
    {
        double theta = -ALPHA * 78;
        api_set_STEERING_control(pca9685, theta);
        cout << "Turn Right==================================================" << endl;
        sleep(1);
        cout << "Normal" << endl;
    }
}

//////////////////////////////////////

=======
>>>>>>> parent of 035dac7... Semi Round
///////// utilitie functions  ///////////////////////////

int main( int argc, char* argv[] ) {
    GPIO *gpio = new GPIO();
    int sw1_stat = 1;
	int sw2_stat = 1;
	int sw3_stat = 1;
	int sw4_stat = 1;
	int sensor = 0;
	
	// Setup input
	gpio->gpioExport(SW1_PIN);
	gpio->gpioExport(SW2_PIN);
	gpio->gpioExport(SW3_PIN);
	gpio->gpioExport(SW4_PIN);
	gpio->gpioExport(SENSOR);
	
	gpio->gpioSetDirection(SW1_PIN, INPUT);
	gpio->gpioSetDirection(SW2_PIN, INPUT);
	gpio->gpioSetDirection(SW3_PIN, INPUT);
	gpio->gpioSetDirection(SW4_PIN, INPUT);
	gpio->gpioSetDirection(SENSOR, INPUT);
    usleep(10000);
    
/// Init openNI ///
    Status rc;
    Device device;

    VideoStream depth, color;
    rc = OpenNI::initialize();
    if (rc != STATUS_OK) {
        printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
        return 0;
    }
    rc = device.open(ANY_DEVICE);
    if (rc != STATUS_OK) {
        printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
        return 0;
    }
    if (device.getSensorInfo(SENSOR_DEPTH) != NULL) {
        rc = depth.create(device, SENSOR_DEPTH);
        if (rc == STATUS_OK) {
            VideoMode depth_mode = depth.getVideoMode();
            depth_mode.setFps(30);
            depth_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
            depth_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);
            depth.setVideoMode(depth_mode);

            rc = depth.start();
            if (rc != STATUS_OK) {
                printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
            }
        }
        else {
            printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
        }
    }

    if (device.getSensorInfo(SENSOR_COLOR) != NULL) {
        rc = color.create(device, SENSOR_COLOR);
        if (rc == STATUS_OK) {
            VideoMode color_mode = color.getVideoMode();
            color_mode.setFps(30);
            color_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
            color_mode.setPixelFormat(PIXEL_FORMAT_RGB888);
            color.setVideoMode(color_mode);

            rc = color.start();
            if (rc != STATUS_OK)
            {
                printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
            }
        }
        else {
            printf("Couldn't create color stream\n%s\n", OpenNI::getExtendedError());
        }
    }
    
  VideoFrameRef frame_depth, frame_color;
    VideoStream* streams[] = {&depth, &color};
/// End of openNI init phase ///
    
/// Init video writer and log files ///   
    bool is_save_file = false; // set is_save_file = true if you want to log video and i2c pwm coeffs.
    VideoWriter depth_videoWriter;	
    VideoWriter color_videoWriter;
    VideoWriter gray_videoWriter;
     
    string gray_filename = "gray.avi";
	string color_filename = "color.avi";
	string depth_filename = "depth.avi";
	
	Mat depthImg, colorImg, grayImage;
	int codec = CV_FOURCC('D','I','V', 'X');
	int video_frame_width = VIDEO_FRAME_WIDTH;
    int video_frame_height = VIDEO_FRAME_HEIGHT;
	Size output_size(video_frame_width, video_frame_height);

   	FILE *thetaLogFile; // File creates log of signal send to pwm control
	if(is_save_file) {
	    gray_videoWriter.open(gray_filename, codec, 8, output_size, false);
        color_videoWriter.open(color_filename, codec, 8, output_size, true);
        //depth_videoWriter.open(depth_filename, codec, 8, output_size, false);
        thetaLogFile = fopen("thetaLog.txt", "w");
	}
/// End of init logs phase ///

    int dir = 0, throttle_val = 0;
    double theta = 0;
    int current_state = 0;
    char key = 0;

    //=========== Init  =======================================================

    ////////  Init PCA9685 driver   ///////////////////////////////////////////

    PCA9685 *pca9685 = new PCA9685() ;
    api_pwm_pca9685_init( pca9685 );

    if (pca9685->error >= 0)
       // api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
        api_set_FORWARD_control( pca9685,throttle_val);
    /////////  Init UART here   ///////////////////////////////////////////////
    /// Init MSAC vanishing point library
    MSAC msac;
    cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,
                            VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);

    api_vanishing_point_init( msac );

    ////////  Init direction and ESC speed  ///////////////////////////
    int set_throttle_val = 0;
    throttle_val = 0;
    theta = 0;
    
    // Argc == 2 eg ./test-autocar 27 means initial throttle is 27
    if(argc == 2 ) set_throttle_val = atoi(argv[1]);
    fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
    int frame_width = VIDEO_FRAME_WIDTH;
    int frame_height = VIDEO_FRAME_HEIGHT;
    Point carPosition(frame_width / 2, frame_height);
    Point prvPosition = carPosition;

    bool running = false, started = false, stopped = false;

    double st = 0, et = 0, fps = 0;
    double freq = getTickFrequency();


    bool is_show_cam = true;
	int count_s,count_ss;
    int frame_id = 0;
<<<<<<< HEAD
    vector<cv::Vec4i> lines;
    bool oneLine = false;
    int preX = 0;
    int preY = 0;
    bool preLeft = false;
    bool preRight = false;

    sign_recognizer sr;
    sr.init();
    /*
    if (TEST_DETECT_SIGN)
    {
        while (1)
        {
            int readyStream = -1;
            rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
            if (rc != STATUS_OK)
            {
                printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
                break;
            }

            depth.readFrame(&frame_depth);
            color.readFrame(&frame_color);
            frame_id++;
            char recordStatus = analyzeFrame(frame_depth, frame_color, depthImg, colorImg);
            flip(depthImg, depthImg, 1);
            flip(colorImg, colorImg, 1);
            Mat img = colorImg.clone();
            resize(img, img, cv::Size(FRAME_HEIGHT, FRAME_WIDTH));
            imshow("img", img);

            int class_id = api_sign_detection(img, sr);

            if (class_id == SIGN_LEFT)
                cout << "=> LEFT SIGN\n";
            else if (class_id == SIGN_RIGHT)
                cout << "=> RIGHT SIGN\n";
            else
                cout << "=> NO SIGN\n";

            waitKey(1);
            //destroyAllWindows();
        }
    }*/

    int road_width;
    bool road_width_set = false;

    while (true)
    {
        Point center_point(0, 0);
=======
	vector<cv::Vec4i> lines;
    while ( true )
    { 
        Point center_point(0,0);
>>>>>>> parent of 035dac7... Semi Round

        st = getTickCount();
        key = getkey();
       unsigned int bt_status = 0;
		gpio->gpioGetValue(SW4_PIN, &bt_status);
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

        if( running )
        {
			//// Check PCA9685 driver ////////////////////////////////////////////
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
            int readyStream = -1;
		    rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
		    if (rc != STATUS_OK)
		    {
		        printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		        break;
		    }

		depth.readFrame(&frame_depth);
		color.readFrame(&frame_color);
            	frame_id ++;
		char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);
		flip(depthImg, depthImg, 1);
		flip(colorImg, colorImg, 1);
		
            ////////// Detect Center Point ////////////////////////////////////
            if (recordStatus == 'c') {
                cvtColor(colorImg, grayImage, CV_BGR2GRAY);
                cv::Mat dst = keepLanes(grayImage, false);
                // cv::imshow("dst", dst);
                cv::Point shift (0, 3 * grayImage.rows / 4);
                bool isRight = true;
                cv::Mat two = twoRightMostLanes(grayImage.size(), dst, shift, isRight);
                // cv::imshow("two", two);
		Rect roi2(0,   3*two.rows / 4, two.cols, two.rows / 4); //c?t ?nh
		
		Mat imgROI2 = two(roi2);
		// cv::imshow("roi", imgROI2);
		int widthSrc = imgROI2.cols;
		int heightSrc = imgROI2.rows;
		vector<Point> pointList;
			//for (int y = 0; y < heightSrc; y++)
			//{
				for (int x = widthSrc; x >= 0; x--)
				{
					if (imgROI2.at<uchar>(25, x) == 255 )
					{
						pointList.push_back(Point(25, x));
						//break;
					}
					if(pointList.size() == 0){
	

							pointList.push_back(Point(20, 300));
							}
				
				}
			//}
			std::cout<<"size"<<pointList.size()<<std::endl;
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
                if (center_point.x == 0 && center_point.y == 0) {
			// modified here: Khong thay duong -> dung
			// previous: Khong thay duong -> dung tam duong cu
			center_point = prvPosition;
			//api_set_STEERING_control(pca9685, 0);
//			api_set_FORWARD_control(pca9685, 0);
			
			// break;
		}
                prvPosition = center_point;
                double angDiff = getTheta(carPosition, Point(xTam, yTam));
		if(-20<angDiff&&angDiff<20)angDiff=0;
                theta = (angDiff*0.8);
		std::cout<<"angdiff"<<angDiff<<std::endl;
		       // theta = (0.00);
		        api_set_STEERING_control(pca9685,theta);
            }
			
            int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
            et = getTickCount();
            fps = 1.0 / ((et-st)/freq);
            cerr << "FPS: "<< fps<< '\n';

            if (recordStatus == 'c' && is_save_file) {
                // 'Center': target point
                // pwm2: STEERING coefficient that pwm at channel 2 (our steering wheel's channel)
                fprintf(thetaLogFile, "Center: [%d, %d]\n", center_point.x, center_point.y);
                fprintf(thetaLogFile, "pwm2: %d\n", pwm2);
                
                if (!colorImg.empty())
			        color_videoWriter.write(colorImg);
			    if (!grayImage.empty())
			        gray_videoWriter.write(grayImage); 
            }
            if (recordStatus == 'd' && is_save_file) {
                if (!depthImg.empty())
                   depth_videoWriter.write(depthImg);
            }

            //////// using for debug stuff  ///////////////////////////////////
            if(is_show_cam) {
                if(!grayImage.empty())
                    imshow( "gray", grayImage );
                waitKey(10);
            }
            if( key == 27 ) break;
        }
        else {
			theta = 0;
            throttle_val = 0;
            if (!stopped) {
                fprintf(stderr, "OFF\n");
<<<<<<< HEAD
                stopped = true;
                started = false;
            }
            api_set_FORWARD_control(pca9685, throttle_val);stderr, "OFF\n");
                stopped = true;
                started = false;
            }
            api_set_FORWARD_control(pca9685, throttle_val);
=======
                stopped = true; started = false;
			}
			api_set_FORWARD_control( pca9685,throttle_val);
>>>>>>> parent of 035dac7... Semi Round
            sleep(1);
        }
    }
    //////////  Release //////////////////////////////////////////////////////
	if(is_save_file)
    {
        gray_videoWriter.release();
        color_videoWriter.release();
        //depth_videoWriter.release();
        fclose(thetaLogFile);
	}
    return 0;
}


