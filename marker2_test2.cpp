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

bool compareLines (Vec2f i, Vec2f j) { return (i[0] > j[0] ); }

int main()
{
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int rho = 0;
    int theta = 0;
    int line_threshold = 100;
    int srn = 0;
    int stn = 0;
    int cannyThreshold1 = 10;
    int cannyThreshold2 = 10;

    int iLowH = 0;
    int iHighH = 255;

    int iLowS = 0;
    int iHighS = 70;

    int iLowV = 40;
    int iHighV = 255;

    int thresholdVal = 95;
    int thresholdMax = 255;
    int thresholdMode = 1;

    int thresh = 73;
    int points = 1000;

    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)

    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);
    cvCreateTrackbar("Threshold value", "Control", &thresholdVal, 500);
    cvCreateTrackbar("Threshold max", "Control", &thresholdMax, 500);
    cvCreateTrackbar("Threshold mode", "Control", &thresholdMode, 4);
    //Create trackbars in "Control" window
    cvCreateTrackbar("Rho", "Control", &rho, 255);
    cvCreateTrackbar("Degrees", "Control", &theta, 180);
    cvCreateTrackbar("Line threshold", "Control", &line_threshold, 255);
    cvCreateTrackbar("srn", "Control", &srn, 100);
    cvCreateTrackbar("stn", "Control", &stn, 100);
    cvCreateTrackbar("Contour thresh", "Control", &thresh, 255);
    cvCreateTrackbar("Contour points", "Control", &points, 1000);

    String folderpath = "../final_project/images/marker_thinline_hard/marker_thinline_*.png";
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

            Mat imgGray, imgOriginalHSV, imgHSV, imgThresholded;

            cvtColor(imgOriginal, imgOriginalHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
            inRange(imgOriginalHSV, Scalar(0, 0, 40), Scalar(255, 70, 255), imgHSV); //Threshold the image

            //morphological opening (remove small objects from the foreground)
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            //morphological closing (fill "small" holes in the foreground)
            dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) );
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_RECT, Size(150, 150)));
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_RECT, Size(100, 100)));
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)));
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)));
            blur( imgHSV, imgHSV, Size(49, 49));
            imshow("HSV1", imgHSV);

            //

         //   inRange(imgOriginal, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

            cvtColor(imgOriginal, imgThresholded, COLOR_BGR2GRAY);
            cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);

            //
            //imshow("Thresholded", imgGray);
            //

            //


            Mat imgHSV_32 = imgHSV.clone();
            Mat imgThresholded_32 = imgGray.clone();
            imgHSV_32.convertTo(imgHSV_32, CV_32FC1);
            imgThresholded_32.convertTo(imgThresholded_32, CV_32FC1);

            Mat imgProduct = imgHSV_32.mul(imgThresholded_32);
            imgProduct.convertTo(imgProduct, CV_32FC1, 1.0f/65025.0f * 255);
            imgProduct.convertTo(imgProduct, CV_8UC1);
            Mat imgCorner = imgProduct.clone();
            //erode(imgProduct, imgProduct, getStructuringElement(MORPH_RECT, Size(20, 20)));
           //
       //     inRange(imgProduct, Scalar(80, 0, 0), Scalar(150, 255, 255), imgProduct); //Threshold the image



            // Reduce the noise
            GaussianBlur( imgProduct, imgProduct, Size(9, 9), 2, 2 );
            imshow("HSV", imgProduct);
            //vector<Vec4i> lines;
            vector<Vec2f> lines;
            Canny(imgProduct, imgProduct, cannyThreshold1, cannyThreshold2);
            imshow( windowName, imgProduct);

            //cvtColor(imgThresholded, imgHSV, COLOR_GRAY2BGR);
            //cvtColor(imgHSV, imgHSV, COLOR_BGR2HSV);



            //Mat imgHSV;// (imgThresholded.size(), CV_8U);
            //inRange(imgThresholded, Scalar(0, 0, 0), Scalar(155, 255, 255), imgHSV); //Threshold the image

            //imshow("HSV", imgHSV);
            //imgHSV.convertTo(imgHSV, CV_8UC1);
            //imgThresholded = imgThresholded(imgHSV);

            //morphological opening (remove small objects from the foreground)
            //
            //dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            //erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
            //imshow("HSV2", imgHSV);


            HoughLines(imgProduct, lines, (double) rho+1, CV_PI / 180 * (double) (theta+1), line_threshold);

            //HoughLinesP(imgProduct, lines, (double) rho+1, CV_PI / 180 * (double) (theta+1), line_threshold, (double) minLineLength, (double) maxLineGap);


            //for( size_t i = 0; i < lines.size(); i++ )
            //{
            //    line( imgOriginal, Point(lines[i][0], lines[i][1]),
            //        Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
            //}


            vector<vector<Point>> p;
            vector<Vec2f> trueThetas;
            Vec2f trueLines(0.0, 0.0);
            for( size_t i = 0; i < lines.size(); i++ )
            {
                vector<float> matches(0);
                matches.push_back(lines[i][1]);
                for (size_t j = 0; j < lines.size(); j++)
                {
                    if(j == i)
                        continue;

                    if (abs(lines[i][1] - lines[j][1]) < 0.1)
                    {
                        matches.push_back(lines[j][1]);
                    }
                    else if(abs(lines[i][1] - lines[j][1] + CV_PI) < 0.1)
                    {
                        if(lines[j][1] < 0.1)
                            matches.push_back(lines[j][1] + CV_PI);
                        else
                            matches.push_back(lines[j][1] - CV_PI);
                    }
                }
                if (matches.size() > 3)
                {
                    float theta = 0;
                    for (size_t k = 0; k < matches.size(); k++)
                    {
                        theta += matches[k];
                    }
                    theta = theta / (float)matches.size();
                    if (theta < 0)
                        theta = theta + CV_PI;
                    else if (theta > CV_PI)
                        theta = theta - CV_PI;
                    trueThetas.push_back(Vec2f((float)matches.size(), theta));
                }
            }

            std::sort(trueThetas.begin(), trueThetas.end(), compareLines);
            while (trueThetas.size() > 2)
            {
                float diff = abs(trueThetas[0][1] - trueThetas[1][1]);
                if(diff < 0.3*CV_PI || diff > 0.7*CV_PI)
                    trueThetas.erase(trueThetas.begin() + 1);
                else
                    trueThetas.erase(trueThetas.begin()+2, trueThetas.end());
            }

            for( size_t i = 0; i < lines.size(); i++ )
            {
                float rho = lines[i][0], theta = lines[i][1];
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                //cout << rho << " " << theta << " " << a << " " << b << " " << x0 << " " << y0 << endl;
                Point p1, p2;
                vector<Point> v;
                p1.x = cvRound(x0 + 1000*(-b));
                p1.y = cvRound(y0 + 1000*(a));
                v.push_back(p1);
                p2.x = cvRound(x0 - 1000*(-b));
                p2.y = cvRound(y0 - 1000*(a));
                v.push_back(p2);
                p.push_back(v);
            }
            vector<Point> intersections;
            for( size_t i = 0; i < p.size(); i++ )
            {
                //line( imgOriginal, p[i][0], p[i][1], Scalar(0,0,255), 3, CV_AA);
                for( size_t j = i+1; j < p.size(); j++)
                {
                    Point d1(p[i][1] - p[i][0]);
                    Point d2(p[j][1] - p[j][0]);
                    Point x(p[j][0] - p[i][0]);

                    double cross = d1.x*d2.y - d1.y*d2.x;
                    if (abs(cross) >= 10e-8)
                    {
                        if (abs(lines[i][1] - lines[j][1]) > 0.3 * CV_PI)
                        {
                            //cout << "Rad: " << abs(lines[i][1] - lines[j][1]) << endl;
                            double t1 = (x.x * d2.y - x.y * d2.x)/cross;
                            Point intersection = p[i][0] + t1*d1;
                            if ( intersection.y > 0 && intersection.x > 0 && (intersection.y < imgHSV.rows - 20) && (intersection.x < imgHSV.cols - 20) && (imgHSV.at<uchar>(intersection.y, intersection.x) != 0) )
                                intersections.push_back(intersection);
                        }


                    }
                }
            }

            vector<Point> trueIntersections;
            vector<int> labels(intersections.size(), 0);
            int label = 1;
            for (size_t i = 0; i < intersections.size(); i++)
            {
                if (labels[i] != 0)
                    continue;
                labels[i] = 1;
                Point intersection1 = intersections[i];
                vector<Point> matches;
                matches.push_back(intersection1);
                for (size_t j = i+1; j < intersections.size(); j++)
                {
                    if(labels[j] != 0)
                        continue;
                    Point intersection2 = intersections[j];
                    int distance = abs(intersection1.x-intersection2.x) + abs(intersection1.y-intersection2.y);
                    if (distance < 20)
                    {
                        labels[j] = 1;
                        matches.push_back(intersection2);
                        //cout << "distance: " << distance << " label: " << labels[j] << endl;
                    }
                }
                if (matches.size() > 1)
                {
                    int x = 0;
                    int y = 0;
                    for (size_t k = 0; k < matches.size(); k++)
                    {
                        Point match = matches[k];
                        x += match.x;
                        y += match.y;
                    }
                    Point trueIntersection(x / matches.size(), y / matches.size());
                    trueIntersections.push_back(trueIntersection);
                }

            }
            //erode(imgGray, imgGray, getStructuringElement(MORPH_RECT, Size(10, 10)));
            threshold(imgGray,imgGray, thresholdVal, thresholdMax, thresholdMode);
            imshow("gray", imgGray);
            cout << "total: " << intersections.size() << endl;
            cout << "true: " << trueIntersections.size() << endl;
            vector<int> mean_vals;
            int old_mean_val;
            for (size_t i = 0; i < trueIntersections.size(); i++)
            {
                Point p = trueIntersections[i];
                //int ROI_size = 6;
                //Rect ROI = Rect(p.x - ROI_size/2, p.y - ROI_size/2, ROI_size, ROI_size);
                //mean_vals.push_back(mean(imgGray(ROI))[0]);
                float mean_val = 0;
                mean_val += imgGray.at<uchar>(p);
                for (size_t j = 0; j < trueThetas.size(); j++)
                {
                    float theta = trueThetas[j][1];// + 0.25*CV_PI;

                    mean_val += imgGray.at<uchar>(p + Point(10 * cos(theta), 10 * sin(theta)));
                    mean_val += imgGray.at<uchar>(p + Point(20 * cos(theta), 20 * sin(theta)));
                    mean_val += imgGray.at<uchar>(p - Point(10 * cos(theta), 10 * sin(theta)));
                    mean_val += imgGray.at<uchar>(p - Point(20 * cos(theta), 20 * sin(theta)));

                    drawMarker(imgOriginal, trueIntersections[i] + Point(20 * cos(theta), 20 * sin(theta)), Scalar(255, 0, 0), MARKER_CROSS);
                    drawMarker(imgOriginal, trueIntersections[i] - Point(20 * cos(theta), 20 * sin(theta)), Scalar(255, 0, 0), MARKER_CROSS);
                }

                mean_vals.push_back(mean_val / 9);
                //
            }
            vector<int> mean_vals_sorted;
            cv::sort(mean_vals, mean_vals_sorted, SORT_ASCENDING);
            for (size_t i = 0; i < trueIntersections.size(); i++)
            {
                if (mean_vals[i] <= mean_vals_sorted[5])
                    drawMarker(imgOriginal, trueIntersections[i], Scalar(0, 255, 0), MARKER_CROSS);
                else

                    drawMarker(imgOriginal, trueIntersections[i], Scalar(0, 0, 255), MARKER_CROSS);
               // drawMarker(imgOriginal, trueIntersections[i] + Point(-10 * sin(0.5 * trueThetas[1][1]), -10 * cos(0.5 * trueThetas[1][1])), Scalar(255, 0, 0), MARKER_CROSS);

            }





//            for( size_t i = 0; i < lines.size(); i++ )
//            {
//               float rho = lines[i][0], theta = lines[i][1];
//               Point p1, p2;
//               double a = cos(theta), b = sin(theta);
//               double x0 = a*rho, y0 = b*rho;
//               p1.x = cvRound(x0 + 1000*(-b));
//               p1.y = cvRound(y0 + 1000*(a));
//               p2.x = cvRound(x0 - 1000*(-b));
//               p2.y = cvRound(y0 - 1000*(a));
//               //line( imgOriginal, p1, p2, Scalar(0,0,255), 3, CV_AA);
//               for( size_t j = i+1; j < lines.size(); j++)
//               {
//                   if(j == 20)
//                       break;
//                   float rho = lines[j][0], theta = lines[j][1];
//                   Point q1, q2;
//                   double a = cos(theta), b = sin(theta);
//                   double x0 = a*rho, y0 = b*rho;
//                   q1.x = cvRound(x0 + 1000*(-b));
//                   q1.y = cvRound(y0 + 1000*(a));
//                   q2.x = cvRound(x0 - 1000*(-b));
//                   q2.y = cvRound(y0 - 1000*(a));
//                   Point d1(p2 - p1);
//                   Point d2(q2 - q1);
//                   Point x(q1 - p1);

//                   double cross = d1.x*d2.y - d1.y*d2.x;
//                   if (abs(cross) >= 10e-8)
//                   {
//                       double t1 = (x.x * d2.y - x.y * d2.x)/cross;
//                       Point intersection = p1 + t1*d1;
//                       circle(imgOriginal, intersection, 5, Scalar(0,255, 0));
//                   }
//               }
//               if(i == 20)
//                   break;
//            }

            imshow("Lines", imgOriginal);

           //
            waitKey();
            break;
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
