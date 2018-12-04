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

}

vector<Point2f> findCircles(const Mat& binaryImg, Mat& outputImg, Scalar color)
{
    Mat blurredImg;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Reduce the noise
    GaussianBlur( binaryImg, blurredImg, Size(9, 9), 2, 2 );

    // find contours
    findContours( blurredImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // get the moments
    vector<Moments> mu(contours.size());
    for( int i = 0; i<contours.size(); i++ )
    {
        mu[i] = moments( contours[i], false );
    }

    // get the centroid of figures.
    vector<Point2f> mc(contours.size());
    for( int i = 0; i<contours.size(); i++)
    {
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }

    int min_area = 3300;
    vector<Point2f> finalPoints;

    // draw contours
    for (std::size_t i = 0; i < contours.size(); i++)
    {
        vector<Point> contours_poly;
        approxPolyDP( Mat(contours[i]), contours_poly, 3, true );
        Rect boundRect = boundingRect( Mat(contours_poly) );
        float radius = (boundRect.height + boundRect.width) / 4;
        float realArea = contourArea(contours[i]);
        float realPerimeter = arcLength(contours[i], true);
        float circleArea = pow(radius, 2) * CV_PI;
        float circlePerimeter = radius * 2 * CV_PI;

        if (realArea < min_area)
            continue;
        if ( abs(realArea - circleArea) > 0.18 * realArea)
            continue;
        if (abs(realPerimeter - circlePerimeter) > 0.2 * realPerimeter)
            continue;
        if (circlePerimeter < 200)
            continue;
        if (circlePerimeter < 215 && abs(realPerimeter - circlePerimeter) > 0.1 * realPerimeter)
            continue;

        //cout << i << ": real perimeter: " << realPerimeter << " circle perimeter: " << circlePerimeter << " - " << contours_poly.size() << " area: " << realArea << endl;

        drawContours(outputImg, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle( outputImg, mc[i], 4, color, -1, 8, 0 );
        rectangle( outputImg, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );
        finalPoints.push_back(mc[i]);

    }
    return finalPoints;
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

            Mat imgOutput = imgOriginal.clone();
            vector<Point2f> redPoint = findCircles(imgBinaryRed, imgOutput, Scalar(0,0,255));
            vector<Point2f> bluePoints = findCircles(imgBinaryBlue, imgOutput, Scalar(255,0,0));
            vector<float> dists;
            vector<float> angles;
            float max_dist = 0;
            size_t index_max_dist;


            for (size_t j = 0; j < bluePoints.size(); j++)
            {
                Point2f diff = redPoint[0]-bluePoints[j];
                float dist = std::sqrt((float)(diff.x*diff.x + diff.y*diff.y));
                dists.push_back(dist);
                float angle = atan2(redPoint[0].y-bluePoints[j].y, redPoint[0].x-bluePoints[j].x);


                angles.push_back(angle);
                if (dist > max_dist)
                {
                    max_dist = dist;
                    index_max_dist = j;
                }
            }


            Point2f bluePointDiagonal = bluePoints[index_max_dist];
            float diagonal_angle = angles[index_max_dist];
            drawMarker(imgOutput, bluePointDiagonal, Scalar(255, 0, 0), MARKER_CROSS);

            bluePoints.erase(bluePoints.begin()+index_max_dist);
            angles.erase(angles.begin()+index_max_dist);
            if(angles[0] - diagonal_angle > angles[1] - diagonal_angle && angles[1] - diagonal_angle)
            {
                drawMarker(imgOutput, bluePoints[0], Scalar(0, 255, 0), MARKER_CROSS);
                drawMarker(imgOutput, bluePoints[1], Scalar(0, 0, 255), MARKER_CROSS);
            }
            else
            {
                drawMarker(imgOutput, bluePoints[1], Scalar(0, 255, 0), MARKER_CROSS);
                drawMarker(imgOutput, bluePoints[0], Scalar(0, 0, 255), MARKER_CROSS);
            }
            cout << "angles: " << i << " " << angles[0] - diagonal_angle << " " << angles[1] - diagonal_angle << endl;
            imshow("Result", imgOutput);
            waitKey();

        }
    }

   return 0;

}
