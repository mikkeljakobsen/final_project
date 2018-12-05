#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "visionHelperFuncs.hpp"

using namespace cv;
using namespace std;

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

vector<Point> sortIntersectionPoints(vector<Point>& candidates)
{
    vector<Vec3f> distances;

    for (size_t n1 = 0; n1 < candidates.size(); n1++)
    {
        Point2f p1 = candidates[n1];

        for (size_t n2 = n1 + 1; n2 < candidates.size(); n2++)
        {
            Point2f p2 = candidates[n2];
            float dist = getDistance(p1, p2);
            size_t index = 0;
            while (index < distances.size())
            {
                if(dist < distances[index][0])
                    break;
                index++;
            }
            distances.emplace(distances.begin()+index, Vec3f(dist, n1, n2));
        }
    }
    vector<Vec2i> minPairs;
    minPairs.emplace_back(Point(distances[0][1], distances[0][2]));
    minPairs.emplace_back(Point(distances[1][1], distances[1][2]));
    vector<Vec2i> linePairs;

    size_t index = 2;
    while (distances.size() > 4)
    {
        Vec3f distancePair = distances[index];
        int n1 = distancePair[1];
        int n2 = distancePair[2];
        if ( !((n1 == minPairs[0][0]) || (n1 == minPairs[0][1]) || (n1 == minPairs[1][0]) || (n1 == minPairs[1][1])) !=
             !((n2 == minPairs[0][0]) || (n2 == minPairs[0][1]) || (n2 == minPairs[1][0]) || (n2 == minPairs[1][1])) )
        {
            if (index >= 4)
            {
                distances.erase(distances.begin()+index, distances.end());
                break;
            }
            size_t i;
            if ((n1 == minPairs[0][0]) || (n1 == minPairs[0][1]) || (n2 == minPairs[0][0]) || (n2 == minPairs[0][1]))
                i = 0;
            else
                i = 1;
            if (n1 != minPairs[i][0] && n1 != minPairs[i][1])
                linePairs.emplace_back(Vec2i(n1, n2));
            else
                linePairs.emplace_back(Vec2i(n2, n1));
            index++;
        }
        else
            distances.erase(distances.begin()+index);
    }


    vector<Point> finalPoints;

    size_t n1 = linePairs[0][0];
    size_t n2 = linePairs[0][1];
    size_t n3 = linePairs[1][0];

    Point2f lineDirection = getDirection(candidates[n1], candidates[n2]);
    Point2f perpendicularDirection = Point2f (-lineDirection.y, lineDirection.x);
    Point2f perpendicularPoint = (Point2f)candidates[n1] + perpendicularDirection * 100;

    if (getDistance(candidates[n3], candidates[n1]) > getDistance(candidates[n3], perpendicularPoint))
    {
        finalPoints.emplace_back(candidates[linePairs[1][0]]);
        finalPoints.emplace_back(candidates[linePairs[1][1]]);
        finalPoints.emplace_back(candidates[linePairs[0][0]]);
        finalPoints.emplace_back(candidates[linePairs[0][1]]);
    }
    else
    {
        finalPoints.emplace_back(candidates[linePairs[0][0]]);
        finalPoints.emplace_back(candidates[linePairs[0][1]]);
        finalPoints.emplace_back(candidates[linePairs[1][0]]);
        finalPoints.emplace_back(candidates[linePairs[1][1]]);

    }

    return finalPoints;
}

vector<Point> findThinLineMarkerPoints(Mat& imgOriginal)
{
    Mat imgGray, imgOriginalHSV, imgMask, imgThresholded;

    cvtColor(imgOriginal, imgOriginalHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    inRange(imgOriginalHSV, Scalar(0, 0, 40), Scalar(255, 70, 255), imgThresholded); //Threshold the image

    // Morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgMask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
    dilate( imgMask, imgMask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
    // Morphological closing (fill "small" holes in the foreground)
    dilate( imgMask, imgMask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);
    erode(imgMask, imgMask, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) , Point(-1,-1), 1, BORDER_CONSTANT, 0);

    // Erode until only one connected component is left
    Mat1i labelImg;
    Mat stats, centroids;
    do
        erode(imgMask, imgMask, getStructuringElement(MORPH_RECT, Size(50, 50)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
    while(connectedComponents(imgMask, labelImg) > 2);
    connectedComponentsWithStats(imgMask, labelImg, stats, centroids);
    int componentArea = stats.at<int>(1, CC_STAT_AREA);

    // Continue eroding until componentArea < 50000
    while(componentArea > 50000)
    {
        erode(imgMask, imgMask, getStructuringElement(MORPH_RECT, Size(10, 10)), Point(-1,-1), 1, BORDER_CONSTANT, 0);
        connectedComponentsWithStats(imgMask, labelImg, stats, centroids);
        componentArea = stats.at<int>(1, CC_STAT_AREA);
    }
    // Dilate to get the whole marker back
    dilate(imgMask, imgMask, getStructuringElement(MORPH_RECT, Size(50, 50)), Point(-1,-1), 2, BORDER_CONSTANT, 0);
    dilate(imgMask, imgMask, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)), Point(-1,-1), 2, BORDER_CONSTANT, 0);

    // Blur the mask a lot so that the masks edges is not detected by Canny
    blur( imgMask, imgMask, Size(99, 99),Point(-1,-1), BORDER_CONSTANT);
    cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);

    // Apply mask on original grayscale image
    Mat imgHSV_32 = imgMask.clone();
    Mat imgGray_32 = imgGray.clone();
    imgHSV_32.convertTo(imgHSV_32, CV_32FC1);
    imgGray_32.convertTo(imgGray_32, CV_32FC1);
    Mat imgProduct = imgHSV_32.mul(imgGray_32);
    imgProduct.convertTo(imgProduct, CV_32FC1, 1.0f/65025.0f * 255);
    imgProduct.convertTo(imgProduct, CV_8UC1);
    Mat imgCorner = imgProduct.clone();

    // Reduce the noise
    GaussianBlur( imgProduct, imgProduct, Size(9, 9), 2, 2 );

    // Dectect edges
    vector<Vec2f> lines;
    Canny(imgProduct, imgProduct, 90, 140);

    // Detect lines
    HoughLines(imgProduct, lines, 1, CV_PI / 180, 100);

    // Find line segments (point-pairs)
    vector<vector<Point>> p;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
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

    // Find intersecting lines
    vector<Point> intersections;
    for( size_t i = 0; i < p.size(); i++ )
    {
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
                    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
                    Point intersection = p[i][0] + t1*d1;
                    if ( intersection.y > 0 && intersection.x > 0 && (intersection.y < imgMask.rows)
                         && (intersection.x < imgMask.cols) && (imgMask.at<uchar>(intersection.y, intersection.x) != 0) )
                        intersections.push_back(intersection);
                }
            }
        }
    }

    // Find true intersections (interpolate between close points)
    vector<Point> trueIntersections;
    vector<int> labels(intersections.size(), 0);
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
            if (distance < 30)
            {
                labels[j] = 1;
                matches.push_back(intersection2);
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
    vector<Point> falseIntersections;

    // Erode until only 6 points is left
    removeOutlierPoints(trueIntersections, falseIntersections, imgMask, 6);

    // Sort points by looking at inter-distances and angles
    return sortIntersectionPoints(trueIntersections);
}

int main()
{

    String folderpath = "../final_project/images/marker_thinline/marker_thinline_*.png";
    vector<String> filenames;
    glob(folderpath, filenames);

    while (true)
    {
        for (size_t i=0; i<filenames.size(); i++)
        {
            while (true){

            Mat imgOriginal = imread(filenames[i], IMREAD_COLOR);


            vector<Point> markerPoints = findThinLineMarkerPoints(imgOriginal);
            vector<Scalar> colors;
            colors.push_back(Scalar(255, 255, 255));
            colors.push_back(Scalar(0, 0, 255));
            colors.push_back(Scalar(0, 255, 0));
            colors.push_back(Scalar(255, 0, 0));


            for (size_t j=0; j < markerPoints.size(); j++)
            {
                drawMarker(imgOriginal, markerPoints[j], colors[j]);
            }
            imshow("Lines", imgOriginal);

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
