#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Point2f getDirection(Point2f a, Point2f b)
{
    Point2f direction = a-b;
    float mag = std::sqrt((float)(direction.x*direction.x + direction.y*direction.y));
    direction.x /= mag;
    direction.y /= mag;
    return direction;
}

float getDistance(Point2f a, Point2f b)
{
    Point2f direction = a-b;
    return std::sqrt((float)(direction.x*direction.x + direction.y*direction.y));
}

float getAngle(Point2f a, Point2f b)
{
    return atan2(a.y-b.y, a.x-b.x);
}
