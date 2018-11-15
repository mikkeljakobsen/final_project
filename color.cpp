#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void redFilter(const Mat& src, Mat& imgThresholded)
{
    Mat imgHSV;
    cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert from BGR to HSV

    inRange(imgHSV, Scalar(0, 140, 80), Scalar(20, 220, 255), imgThresholded); //Threshold the image

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );

    // Reduce the noise
    GaussianBlur( imgThresholded, imgThresholded, Size(9, 9), 2, 2 );
}

void blueFilter(const Mat& src, Mat& imgThresholded)
{
    Mat imgHSV;
    cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert from BGR to HSV

    inRange(imgHSV, Scalar(110, 70, 35), Scalar(130, 180, 180), imgThresholded); //Threshold the image

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );

    // Reduce the noise
    GaussianBlur( imgThresholded, imgThresholded, Size(9, 9), 2, 2 );
}


 int main()
 {
    String folderpath = "../final_project/images/marker_color_hard/marker_color_*.png";
    vector<String> filenames;
    glob(folderpath, filenames);

    while (true)
    {
        for (size_t i=0; i<filenames.size(); i++)
        {
            Mat imgOriginal = imread(filenames[i], IMREAD_COLOR);
            Mat imgBinaryRed, imgBinaryBlue;

            redFilter(imgOriginal, imgBinaryRed);
            blueFilter(imgOriginal, imgBinaryBlue);
            imshow("Red", imgBinaryRed);
            imshow("Blue", imgBinaryBlue);

            vector<Vec3f> redCircle, blueCircles;
            HoughCircles( imgBinaryRed, redCircle, HOUGH_GRADIENT, 1, imgBinaryRed.rows, 50, 15, 30, 60 );
            HoughCircles( imgBinaryBlue, blueCircles, HOUGH_GRADIENT, 1, imgBinaryBlue.rows/12, 80, 17, 30, 60 );

            Mat imgOutput = imgOriginal.clone();

            for( size_t j = 0; j < redCircle.size(); j++ )
            {
                Point center(cvRound(redCircle[j][0]), cvRound(redCircle[j][1]));
                int radius = cvRound(redCircle[j][2]);
                // circle center
                circle( imgOutput, center, 3, Scalar(0,255,0), -1, 8, 0 );
                // circle outline
                circle( imgOutput, center, radius, Scalar(0,0,255), 3, 8, 0 );
            }

            for( size_t j = 0; j < blueCircles.size(); j++ )
            {
                Point center(cvRound(blueCircles[j][0]), cvRound(blueCircles[j][1]));
                int radius = cvRound(blueCircles[j][2]);
                // circle center
                circle( imgOutput, center, 3, Scalar(0,255,0), -1, 8, 0 );
                // circle outline
                circle( imgOutput, center, radius, Scalar(255,0,0), 3, 8, 0 );
            }
            imshow("Result", imgOutput);

            if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                return 0;
            }
        }
    }

   return 0;

}
