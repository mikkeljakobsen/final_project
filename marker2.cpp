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

void removeOutlierPoints(vector<Point>& truePoints, vector<Point>& falsePoints, Mat& img, size_t max_size)
{
    while(truePoints.size() > max_size)
    {
        erode(img, img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
        for (size_t i = 0; i < truePoints.size() && truePoints.size() > max_size;)
        {
            if(img.at<uchar>(truePoints[i]) < 100)
            {
                falsePoints.push_back(truePoints[i]);
                truePoints.erase(truePoints.begin()+i);

            }
            else
                i++;
        }
    }

}

bool compareLines (Vec2f i, Vec2f j) { return (i[0] > j[0] ); }
bool sortDescending (size_t i, size_t j) { return (i > j ); }

int main()
{
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int rho = 0;
    int theta = 0;
    int line_threshold = 100;
    int srn = 0;
    int stn = 0;
    int cannyThreshold1 = 90;
    int cannyThreshold2 = 140;

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

    cvCreateTrackbar("LowS", "Control", &cannyThreshold1, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &cannyThreshold2, 255);

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
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
            dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
            //morphological closing (fill "small" holes in the foreground)
            dilate( imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
            erode(imgHSV, imgHSV, getStructuringElement(MORPH_RECT, Size(150, 150)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_RECT, Size(100, 100)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
            dilate(imgHSV, imgHSV, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
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
            vector<Vec2f> trueLines;
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
                            matches.push_back(lines[j][1] - CV_PI);
                    else if(abs(lines[i][1] - lines[j][1] - CV_PI) < 0.1)
                            matches.push_back(lines[j][1] + CV_PI);

                }
                if (matches.size() > 1)
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
                if(diff < 0.05*CV_PI || diff > 0.95*CV_PI)
                    trueThetas.erase(trueThetas.begin() + 1);
                else
                    trueThetas.erase(trueThetas.begin()+2, trueThetas.end());
            }
           /* for( size_t k = 0; k < trueThetas.size(); k++ )
            {
                vector<Vec2f> trueRhos;
                for (size_t i = 0; i < lines.size(); i++)
                {
                    if (abs(lines[i][1] - trueThetas[k][1]) > 0.1
                            && abs(lines[i][1] - trueThetas[k][1] + CV_PI) > 0.1
                            && abs(lines[i][1] - trueThetas[k][1] - CV_PI) > 0.1)
                        continue;

                    vector<float> matches(0);
                    matches.push_back(lines[i][0]);
                    for (size_t j = 0; j < lines.size(); j++)
                    {
                        if(j == i)
                            continue;
                        if (abs(lines[j][1] - trueThetas[k][1]) > 0.1
                                && abs(lines[j][1] - trueThetas[k][1] + CV_PI) > 0.1
                                && abs(lines[j][1] - trueThetas[k][1] - CV_PI) > 0.1)
                            continue;

                        if (abs(lines[i][0] - lines[j][0]) < 20)
                            matches.push_back(lines[j][0]);

                    }
                    if (matches.size() > 1)
                    {
                        float rho = 0;
                        for (size_t k = 0; k < matches.size(); k++)
                        {
                            rho += matches[k];
                        }
                        rho = rho / (float)matches.size();
                        trueRhos.push_back(Vec2f((float)matches.size(), rho));
                    }
                }
                std::sort(trueRhos.begin(), trueRhos.end(), compareLines);
                size_t element = 0;
                while (trueRhos.size() > 3)
                {
                    if (element == 0 && abs(trueRhos[0][1] - trueRhos[1][1]) < 20)
                        trueRhos.erase(trueRhos.begin() + 1);
                    else if(element == 1 && (abs(trueRhos[0][1] - trueRhos[2][1]) < 20 || abs(trueRhos[1][1] - trueRhos[2][1]) < 20) )
                        trueRhos.erase(trueRhos.begin() + 2);
                    else if(element == 2)
                        trueRhos.erase(trueRhos.begin() + 3, trueRhos.end());
                    else
                        element++;
                }
                float theta = trueThetas[k][1];
                cout << "theta: " << theta << " size " << trueRhos.size() << endl;
                for (size_t n = 0; n < trueRhos.size(); n++)
                {
                    float rho = trueRhos[n][1];
                    cout << " rho: " << rho << endl;

                    double a = cos(theta), b = sin(theta);
                    double x0 = a*rho, y0 = b*rho;
                    //cout << rho << " " << theta << " " << a << " " << b << " " << x0 << " " << y0 << endl;
                    Point p1, p2;
                    p1.x = cvRound(x0 + 1000*(-b));
                    p1.y = cvRound(y0 + 1000*(a));
                    p2.x = cvRound(x0 - 1000*(-b));
                    p2.y = cvRound(y0 - 1000*(a));
                    line( imgOriginal, p1, p2, Scalar(0,0,255), 3, CV_AA);
                }
            }*/
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
                        if (abs(lines[i][1] - lines[j][1]) > 0.3 * CV_PI && abs(lines[i][1] - lines[j][1]) < 0.7 * CV_PI)
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
                    x /= matches.size();
                    y /= matches.size();
                    trueIntersections.push_back(Point(x, y));
                }

            }



            //erode(imgGray, imgGray, getStructuringElement(MORPH_RECT, Size(10, 10)));
            //threshold(imgGray,imgGray, thresholdVal, thresholdMax, thresholdMode);
            //imshow("gray", imgGray);
            //cout << "total: " << intersections.size() << endl;
            //cout << "true: " << trueIntersections.size() << endl;
            vector<float> mean_vals;
            int old_mean_val;

            /*for(size_t i = 0; i < trueThetas.size(); i++)
            {
                float rho = 500, theta = trueThetas[i][1];
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                Point p1, p2;
                p1.x = cvRound(x0 + 1000*(-b));
                p1.y = cvRound(y0 + 1000*(a));
                p2.x = cvRound(x0 - 1000*(-b));
                p2.y = cvRound(y0 - 1000*(a));
                line( imgOriginal, p1, p2, Scalar(0,0,255), 3, CV_AA);
            }*/
            vector<Point> falseIntersections;
            removeOutlierPoints(trueIntersections, falseIntersections, imgHSV, 6);
/*
            for (size_t i = 0; i < falseIntersections.size(); i++)
            {
                vector<Point> contours_poly;
                approxPolyDP( Mat(trueIntersections), contours_poly, 3, true );
                if (contours_poly.size() <= 5)
                    break;
                cout << "poly_size: " << contours_poly.size() << endl;
                waitKey();
                Point falseNegative = falseIntersections[i];
                removeOutlierPoints(trueIntersections, falseIntersections, imgHSV, 6-i);
                trueIntersections.push_back(falseNegative);

            }
*/


            vector<float> distances;
            for (size_t n1 = 0; n1 < trueIntersections.size(); n1++)
            {
                Point p1 = trueIntersections[n1];

                for (size_t n2 = n1 + 1; n2 < trueIntersections.size(); n2++)
                {
                    Point p2 = trueIntersections[n2];
                    Point diff = p2-p1;
                    float dist = std::sqrt((float)(diff.x*diff.x + diff.y*diff.y));
                    distances.push_back(dist);
                }
            }


            vector<Point> finalPoints;
            vector<size_t> indices;
            vector<float> distances_sorted;
            cv::sort(distances, distances_sorted, SORT_ASCENDING);
            size_t index = 0;
            for (size_t n1 = 0; n1 < trueIntersections.size(); n1++)
            {
                Point p1 = trueIntersections[n1];

                for (size_t n2 = n1 + 1; n2 < trueIntersections.size(); n2++)
                {
                    Point p2 = trueIntersections[n2];
                    if (distances[index++] <= distances_sorted[1])
                    {
                        indices.push_back(n1);
                        finalPoints.push_back(p1);
                        indices.push_back(n2);
                        finalPoints.push_back(p2);
                        //drawMarker(imgOriginal, p1, Scalar(0, 255, 0), MARKER_CROSS);
                        //drawMarker(imgOriginal, p2, Scalar(0, 0, 255), MARKER_CROSS);
                    }
                }
            }
            std::sort(indices.begin(), indices.end(), sortDescending);
            for (size_t i = 0; i < indices.size(); i++)
            {
                cout << indices[i] << endl;
                trueIntersections.erase(trueIntersections.begin()+indices[i]);
            }

            Point p1_final, p2_final, p3_final, p4_final;

            for(size_t i = 0; i < 2; i++)
            {
                Point p1 = trueIntersections[i];
                Point p2 = finalPoints[2*i];
                Point p3 = finalPoints[2*i+1];
                float angle1 = atan2(p1.y-p2.y, p1.x-p2.x);
                float angle2 = atan2(p1.y-p3.y, p1.x-p3.x);
                if (abs(abs(angle1)-abs(angle2)) > 0.1 * CV_PI)
                    continue;
                p1_final = p1;
                LineIterator it1(imgGray, p1, p2);
                LineIterator it2(imgGray, p1, p3);
                if (it1.count > it2.count)
                    p2_final = p2;
                else
                    p2_final = p3;
                i++;
                if (i == 2)
                    i = 0;
                Point p4 = trueIntersections[i];
                Point p5 = finalPoints[2*i];
                Point p6 = finalPoints[2*i+1];
                float angle3 = atan2(p4.y-p5.y, p4.x-p5.x);
                float angle4 = atan2(p4.y-p6.y, p4.x-p6.x);
                if (abs(abs(angle3)-abs(angle4)) > 0.1 * CV_PI)
                    break;
                p4_final = p4;
                LineIterator it3(imgGray, p4, p5);
                LineIterator it4(imgGray, p4, p6);
                if (it3.count > it4.count)
                    p3_final = p5;
                else
                    p3_final = p6;
                break;
            }


            drawMarker(imgOriginal, p1_final, Scalar(255, 255, 255), MARKER_CROSS);
            drawMarker(imgOriginal, p2_final, Scalar(0, 255, 0), MARKER_CROSS);
            drawMarker(imgOriginal, p3_final, Scalar(0, 255, 255), MARKER_CROSS);
            drawMarker(imgOriginal, p4_final, Scalar(0, 0, 255), MARKER_CROSS);

            imshow("Lines", imgOriginal);

            /*
            for (size_t n1 = 0; n1 < trueIntersections.size(); n1++)
            {
                Point p1 = trueIntersections[n1];

                for (size_t n2 = 0; n2 < trueIntersections.size(); n2++)
                {
                    if (n2 == n1)
                        continue;
                    Point p2 = trueIntersections[n2];

                    LineIterator it1(imgGray, p1, p2);
                    float angle1 = atan2(p2.y-p1.y, p2.x-p1.x);

                    for (size_t n3 = 0; n3 < trueIntersections.size(); n3++)
                    {
                        if (n3 == n2 || n3 == n1)
                            continue;
                        Point p3 = trueIntersections[n3];
                        double s_line1_line2 = (p2-p1).dot(p3-p2);
                       // if (s_line1_line2 < 0.95 or s_line1_line2 > 1.05)
                         //   continue;

                        LineIterator it2(imgGray, p2, p3);
                        float angle2 = atan2(p3.y-p2.y, p3.x-p2.x);

                        if (it1.count / it2.count < 4 || it1.count / it2.count > 6)
                            continue;
                        if (abs(angle2-angle1) > 0.1 * CV_PI)// && abs(angle2-angle1) < 0.99 * CV_PI)
                            continue;

                        for (size_t n4 = 0; n4 < trueIntersections.size(); n4++)
                        {
                            if(n4 == n3 || n4 == n2 || n4 == n1)
                                continue;
                            Point p4 = trueIntersections[n4];
                            double s_line2_line3 = (p3-p2).dot(p4-p3);
                            LineIterator it3(imgGray, p3, p4);
                            float angle3 = atan2(p4.y-p3.y, p4.x-p3.x);

                            //if (abs(angle3-angle2) < 0.3 * CV_PI || abs(angle3-angle2) > 0.7 * CV_PI)
                              //  continue;

                            for (size_t n5 = 0; n5 < trueIntersections.size(); n5++)
                            {
                                if (n5 == n4 || n5 == n3 || n5 == n2 || n5 == n1)
                                    continue;
                                Point p5 = trueIntersections[n5];
                                double s_line3_line4 = (p4-p3).dot(p5-p4);
                                LineIterator it4(imgGray, p4, p5);
                                float angle4 = atan2(p4.y-p5.y, p4.x-p5.x);

                                //if (abs(it4.count - it2.count) > 5)
                                 //   continue;
                               // if (abs(angle4-angle3) < 0.3 * CV_PI || abs(angle4-angle3) > 0.7 * CV_PI)
                                 //   continue;
                                if (abs(angle4-angle2) > 0.1 * CV_PI)// && abs(angle4-angle2) < 0.99 * CV_PI)
                                    continue;
                                //if (s_line1_line2 < 0 && s_line3_line4 > 0 || s_line1_line2 > 0 && s_line3_line4 < 0)
                                  //  continue;

                                for (size_t n6 = 0; n6 < trueIntersections.size(); n6++)
                                {
                                    if (n6 == n5 || n6 == n4 || n6 == n3 || n6 == n2 || n6 == n1)
                                        continue;
                                    Point p6 = trueIntersections[n6];

                                    double s_line4_line5 = (p5-p4).dot(p6-p5);

                                    LineIterator it5(imgGray, p5, p6);
                                    float angle5 = atan2(p5.y-p6.y, p5.x-p6.x);

                                    if (abs(angle5-angle4) > 0.1 * CV_PI)// && abs(angle5-angle4) < 0.99 * CV_PI)
                                    {
                                        cout << "angle4 " << angle4 << " angle5 " << angle5 << endl;
                                        continue;
                                    }
                                    //if (it5.count / it4.count < 4 || it5.count / it4.count > 6)
                                    //{
                                     //   cout << "fail " << it4.count << " " << it5.count << endl;
                                      //  continue;
                                    //}
                                    //if (abs(it5.count - it1.count) > 50)
                                      //  continue;

                                   float angle6 = atan2(p1.y-p6.y, p1.x-p6.x);
                                    double s_line5_line6 = (p6-p5).dot(p1-p6);

                                    double s_line6_line1 = (p1-p6).dot(p2-p1);
                                    LineIterator it6(imgGray, p6, p1);
                                    //if (abs(it6.count-it3.count) > 50)
                                      //  continue;
                               //     if (abs(angle6-angle5) < 0.3 * CV_PI || abs(angle6-angle5) > 0.7 * CV_PI)
                                 //       continue;
                                    //if (s_line1_line2 < 0 && s_line6_line1 < 0 || s_line1_line2 > 0 && s_line6_line1 > 0)
                                    //    continue;
                                    cout << "Solution: " << "p1: " << p1 << "angle: " << angle1 << "l: " << it1.count << "p2: " << p2 << "angle: " << angle2 << "l: " << it2.count
                                         << "p3: " << p3 << "angle: " << angle3 << "l: " << it3.count << "p4: " << p4 << "angle: " << angle4 << "l: " << it4.count
                                         << "p5: " << p5 << "angle: " << angle5 << "l: " << it5.count << "p6: " << p6 << "angle: " << angle6 << "l: " << it6.count << endl;
                                    drawMarker(imgOriginal, p1, Scalar(0, 255, 0), MARKER_CROSS);
                                    drawMarker(imgOriginal, p2, Scalar(0, 0, 255), MARKER_CROSS);
                                    drawMarker(imgOriginal, p3, Scalar(255, 0, 0), MARKER_CROSS);
                                    drawMarker(imgOriginal, p4, Scalar(150, 255, 150), MARKER_CROSS);
                                    drawMarker(imgOriginal, p5, Scalar(200, 255, 200), MARKER_CROSS);
                                    drawMarker(imgOriginal, p6, Scalar(255, 255, 255), MARKER_CROSS);
                                    cout << "Test " << s_line1_line2 << " " << s_line2_line3 << " " << s_line3_line4 << " " << s_line4_line5 << " " << s_line5_line6 << " " << s_line6_line1 << endl;
                                    cout << "angle1: " << angle1 << " angle2 " << angle2 << endl;
                                    imshow("Lines", imgOriginal);
                                    waitKey();

                                }



                            }
                        }
                    }
                }
            }*/
           /* for (size_t i = 0; i < trueIntersections.size(); i++)
            {
                Point p = trueIntersections[i];
                if(imgGray.at<uchar>(p) > 100)
                    continue;
                for (size_t j = 0; j < trueIntersections.size(); j++)
                {
                    if (j == i)
                        continue;
                    Point q = trueIntersections[j];
                    if(imgGray.at<uchar>(q) > 100)
                        continue;
                    float angle1 = atan2(p.y-q.y, p.x-q.x);

                    LineIterator it(imgGray, p, q);
                    for(int n=0; n<it.count && n<10; n++, it++)
                    {
                        uchar val = imgGray.at<uchar>(it.pos()); //mean(imgGray(Rect(it.pos()-Point(3,3), Size(5,5))))[0];
                        if (val > 100)
                            continue;
                        drawMarker(imgOriginal, p, Scalar(255, 0, 0), MARKER_SQUARE, 1);
                    }


                    for (size_t k = 0; k < trueIntersections.size(); k++)
                    {
                        if (k == j || k == i)
                            continue;
                        Point r = trueIntersections[k];
                        if(imgGray.at<uchar>(r) > 100)
                            continue;
                        float angle2 = atan2(p.y-r.y, p.x-r.x);
                        if (abs(angle1-angle2) < 0.3 * CV_PI || abs(angle1-angle2) > 0.7 * CV_PI)
                            continue;
                        LineIterator it(imgGray, p, r);
                        for(int n=0; n<it.count && n<10; n++, it++)

                        {
                            uchar val = imgGray.at<uchar>(it.pos());//mean(imgGray(Rect(it.pos()-Point(3,3), Size(5,5))))[0];
                            if (val > 100)
                                continue;
                            drawMarker(imgOriginal, p, Scalar(255, 0, 0), MARKER_SQUARE, 1);
                        }
                        drawMarker(imgOriginal, p, Scalar(0, 255, 0), MARKER_CROSS);


                    }

                    //mean_val /= it.count;
                    //cout << "Mean val on line from " << trueIntersections[i] << " to " << trueIntersections[j] << ": " << mean_val << endl;
                    //mean_vals.push_back(mean_val);
                }
            }*/
            /*
                Point p = trueIntersections[i];
                //mean_vals.push_back(mean(imgGray(Rect(p.x - 5, p.y - 5, 10, 10)))[0]);
                float mean_val_black = 0;
                float mean_val_white = 0;

                //mean_val_black += 10 * mean(imgGray(ROI))[0];
                mean_val_black += imgGray.at<uchar>(p);
                for (size_t j = 0; j < trueThetas.size(); j++)
                {
                    float theta_line = trueThetas[j][1];
                    float theta_off_line = trueThetas[j][1] + 0.25*CV_PI;


                    mean_val_black += imgGray.at<uchar>(p + Point(10 * cos(theta_line), 10 * sin(theta_line)));
                    mean_val_black += imgGray.at<uchar>(p + Point(20 * cos(theta_line), 20 * sin(theta_line)));
                    mean_val_black += imgGray.at<uchar>(p - Point(10 * cos(theta_line), 10 * sin(theta_line)));
                    mean_val_black += imgGray.at<uchar>(p - Point(20 * cos(theta_line), 20 * sin(theta_line)));
                    mean_val_white += imgGray.at<uchar>(p + Point(10 * cos(theta_off_line), 10 * sin(theta_off_line)));
                    mean_val_white += imgGray.at<uchar>(p + Point(20 * cos(theta_off_line), 20 * sin(theta_off_line)));
                    mean_val_white += imgGray.at<uchar>(p - Point(10 * cos(theta_off_line), 10 * sin(theta_off_line)));
                    mean_val_white += imgGray.at<uchar>(p - Point(20 * cos(theta_off_line), 20 * sin(theta_off_line)));

                    drawMarker(imgOriginal, trueIntersections[i] + Point(20 * cos(theta_line), 20 * sin(theta_line)), Scalar(255, 0, 0), MARKER_CROSS);
                    drawMarker(imgOriginal, trueIntersections[i] - Point(20 * cos(theta_line), 20 * sin(theta_line)), Scalar(255, 0, 0), MARKER_CROSS);
                }

                mean_vals.push_back(abs(mean_val_white-mean_val_black));
                //
            }*/
            //vector<int> votes(trueIntersections.size(), 0);
           // vector<float> mean_vals_sorted;
            //cv::sort(mean_vals, mean_vals_sorted, SORT_ASCENDING);
            /*for (size_t i = 0; i < trueIntersections.size(); i++)
            {
                for (size_t j = i + 1; j < trueIntersections.size(); j++)
                {
                    if (mean_vals[i+j] <= mean_vals_sorted[10])
                    {
                        votes[i] += 1;
                        votes[j] += 1;

                        //drawMarker(imgOriginal, trueIntersections[j], Scalar(0, 255, 0), MARKER_CROSS);
                    }
                    //else
                      //  drawMarker(imgOriginal, trueIntersections[j], Scalar(0, 0, 255), MARKER_CROSS);
                }
                if (votes[i] > 2)
                    drawMarker(imgOriginal, trueIntersections[i], Scalar(0, 255, 0), MARKER_CROSS);
            }
            for (size_t i = 0; i < votes.size(); i++)
                cout << i << ": " << votes[i] << endl;

*/
 /*           for (size_t i = 0; i < trueIntersections.size(); i++)
            {
                if (mean_vals[i] >= mean_vals_sorted[5])
                    drawMarker(imgOriginal, trueIntersections[i], Scalar(0, 255, 0), MARKER_CROSS);
                else
                    drawMarker(imgOriginal, trueIntersections[i], Scalar(0, 0, 255), MARKER_CROSS);
               // drawMarker(imgOriginal, trueIntersections[i] + Point(-10 * sin(0.5 * trueThetas[1][1]), -10 * cos(0.5 * trueThetas[1][1])), Scalar(255, 0, 0), MARKER_CROSS);

            }*/





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

            imshow("HSV2", imgHSV);

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
