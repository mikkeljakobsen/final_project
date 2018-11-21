#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace
{
    // windows and trackbars name
    const std::string windowName = "Hough Circle Detection Demo";
    const std::string cannyThresholdTrackbarName = "Canny threshold";
    const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
    const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

    // initial and max values of the parameters of interests.
    const int cannyThresholdInitialValue = 80;
    const int accumulatorThresholdInitialValue = 17;
    const int maxAccumulatorThreshold = 50;
    const int maxCannyThreshold = 255;

    void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold)
    {
        // will hold the results of the detection
        std::vector<Vec3f> circles;
        // runs the actual detection
        HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/12, cannyThreshold, accumulatorThreshold, 30, 60);

        // clone the colour, input image for displaying purposes
        Mat display = src_display.clone();
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( display, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( display, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }

        // shows the results
        imshow( windowName, display);
    }
}

 int main()
 {
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int rho = 1;
    int theta = 1;
    int line_threshold = 0;
    int minLineLength = 180;
    int maxLineGap = 0;

    //Create trackbars in "Control" window
    cvCreateTrackbar("Rho", "Control", &rho, 255);
    cvCreateTrackbar("Degrees", "Control", &theta, 180);
    cvCreateTrackbar("Line threshold", "Control", &line_threshold, 255);
    cvCreateTrackbar("Min line length", "Control", &minLineLength, 1000);
    cvCreateTrackbar("Max line length", "Control", &maxLineGap, 255);

    String folderpath = "../final_project/images/marker_thinline_hard/marker_thinline_hard_*.png";
    vector<String> filenames;
    glob(folderpath, filenames);

    // create the main window, and attach the trackbars
    namedWindow( windowName, WINDOW_AUTOSIZE );

    while (true)
    {
        for (size_t i=0; i<filenames.size(); i++)
        {
            while (true){

            Mat imgOriginal = imread(filenames[i], IMREAD_COLOR);

            Mat imgGray;

            cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY); //Convert the captured frame from BGR to HSV

            Mat imgThresholded;

            threshold(imgGray,imgThresholded, 170, 255, 2);
            //cvtColor(imgThresholded, imgHSV, COLOR_GRAY2BGR);
            //cvtColor(imgHSV, imgHSV, COLOR_BGR2HSV);


            //morphological closing (fill small holes in the foreground)
            //dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            //erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );


            Mat imgHSV;// (imgThresholded.size(), CV_8U);
            inRange(imgThresholded, Scalar(0, 0, 0), Scalar(155, 255, 255), imgHSV); //Threshold the image
            //imshow("HSV", imgHSV);
            //imgHSV.convertTo(imgHSV, CV_8UC1);
            //imgThresholded = imgThresholded(imgHSV);

            //morphological opening (remove small objects from the foreground)
            //
            //dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            //erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            //imshow("HSV2", imgHSV);
            Mat imgHSV_32 = imgHSV.clone();
            Mat imgThresholded_32 = imgThresholded.clone();
            imgHSV_32.convertTo(imgHSV_32, CV_32FC1);
            imgThresholded_32.convertTo(imgThresholded_32, CV_32FC1);

            Mat imgProduct = imgHSV_32.mul(imgThresholded_32);
            imgProduct.convertTo(imgProduct, CV_32FC1, 1.0f/65025.0f * 255);
            imgProduct.convertTo(imgProduct, CV_8UC1);

            inRange(imgProduct, Scalar(80, 0, 0), Scalar(150, 255, 255), imgProduct); //Threshold the image

            imshow( windowName, imgProduct);

            // Reduce the noise so we avoid false circle detection
            GaussianBlur( imgProduct, imgProduct, Size(9, 9), 2, 2 );
            vector<Vec4i> lines;

            HoughLinesP(imgProduct, lines, rho, CV_PI / 180 * (double) theta, line_threshold, minLineLength, maxLineGap);
            for( size_t i = 0; i < lines.size(); i++ )
            {
                line( imgOriginal, Point(lines[i][0], lines[i][1]),
                    Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
            }


            imshow("Lines", imgOriginal);


            if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                break;
            }
            }
        }
    }

   return 0;

}
