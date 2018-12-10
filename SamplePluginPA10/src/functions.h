//
// Created by quist on 12/3/18.
//
#pragma once

#include <vector>

// RobWork includes
#include <rw/models/WorkCell.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/kinematics.hpp>
#include <rwlibs/opengl/RenderImage.hpp>
#include <rwlibs/simulation/GLFrameGrabber.hpp>

// RobWorkStudio includes
#include <RobWorkStudioConfig.hpp> // For RWS_USE_QT5 definition
#include <rws/RobWorkStudioPlugin.hpp>

#include <rw/math/LinearAlgebra.hpp>

using namespace std;
using namespace rw::common;
using namespace rw::math;
using namespace rw::graphics;
using namespace rw::kinematics;
using namespace rw::models;
using namespace rw::sensor;
using namespace rwlibs::opengl;
using namespace rwlibs::simulation;

vector<Transform3D<double>> loadMarkers(string path)
{
    vector<Transform3D<double>> tempVector;
    // Load The Markers
    ifstream infile(path);
    double x, y, z, roll, pitch, yaw;
    while (infile >> x >> y >> z >> roll >> pitch >> yaw)
    {
        Vector3D<> pos(x, y, z);
        RPY<> rpy(roll, pitch, yaw);
        tempVector.emplace_back(Transform3D<> (pos, rpy.toRotation3D()));
    }
    return tempVector;
}

vector<double> calcUV(MovableFrame* markerFrame, Frame* cameraFrame, vector<Vector3D<double>> posInMarkerFrame, State state, double focalLength, double z)
{
    vector<double> uv;
    for (int i = 0; i < posInMarkerFrame.size(); i++)
    {
        rw::math::Vector3D<double> markerPosInCamFrame = cameraFrame->fTf(markerFrame, state)*posInMarkerFrame[i];
        uv.push_back((focalLength*markerPosInCamFrame[0])/z);
        uv.push_back((focalLength*markerPosInCamFrame[1])/z);
    }

    return uv;
}

Eigen::Matrix<double, 6, 6> calcImgJ(vector<double> uv, double z, double f)
{
    Eigen::Matrix<double, 6, 6> J_img;

    for (int i = 0; i < 5; i = i+2)
    {
        double u = uv[i];
        double v = uv[i+1];
        J_img(i,0) = -f/z;
        J_img(i,1) = 0.0;
        J_img(i,2) = u/z;
        J_img(i,3) = (u*v)/f;
        J_img(i,4) = -(f*f+u*u)/f;
        J_img(i,5) = v;
        J_img(i+1,0) = 0.0;
        J_img(i+1,1) = -f/z;
        J_img(i+1,2) = v/z;
        J_img(i+1,3) = (f*f+v*v)/f;
        J_img(i+1,4) = -(u*v)/f;
        J_img(i+1,5) = -u;
    }

    return J_img;
}


Eigen::Matrix<double, 8, 6> calcImgJVision(vector<double> uv, double z, double f)
{
    Eigen::Matrix<double, 8, 6> J_img;

    for (int i = 0; i < 7; i = i+2)
    {
        double u = uv[i];
        double v = uv[i+1];
        J_img(i,0) = -f/z;
        J_img(i,1) = 0.0;
        J_img(i,2) = u/z;
        J_img(i,3) = (u*v)/f;
        J_img(i,4) = -(f*f+u*u)/f;
        J_img(i,5) = v;
        J_img(i+1,0) = 0.0;
        J_img(i+1,1) = -f/z;
        J_img(i+1,2) = v/z;
        J_img(i+1,3) = (f*f+v*v)/f;
        J_img(i+1,4) = -(u*v)/f;
        J_img(i+1,5) = -u;
    }

    return J_img;
}

Eigen::Matrix<double, 2, 6> calcImgJSingle(vector<double> uv, double z, double f)
{
    Eigen::Matrix<double, 2, 6> J_img;

    for (int i = 0; i < 1; i = i+2)
    {
        double u = uv[i];
        double v = uv[i+1];
        J_img(i,0) = -f/z;
        J_img(i,1) = 0.0;
        J_img(i,2) = u/z;
        J_img(i,3) = (u*v)/f;
        J_img(i,4) = -(f*f+u*u)/f;
        J_img(i,5) = v;
        J_img(i+1,0) = 0.0;
        J_img(i+1,1) = -f/z;
        J_img(i+1,2) = v/z;
        J_img(i+1,3) = (f*f+v*v)/f;
        J_img(i+1,4) = -(u*v)/f;
        J_img(i+1,5) = -u;
    }

    return J_img;
}

Eigen::Matrix<double, 6, 6> calcS(const rw::models::Device::Ptr device, State state, Frame* cameraFrame)
{
    Rotation3D<double> baseCam_R = device->baseTframe(cameraFrame,state).R();
    Eigen::Matrix<double, 3, 3> R = baseCam_R.e();
    Eigen::Matrix<double, 3, 3> RT = R.transpose();

    Eigen::Matrix<double, 6, 6> S;

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            S(i,j) = 0;

    // top left 3x3
    S(0,0) = RT(0,0);
    S(0,1) = RT(0,1);
    S(0,2) = RT(0,2);
    S(1,0) = RT(1,0);
    S(1,1) = RT(1,1);
    S(1,2) = RT(1,2);
    S(2,0) = RT(2,0);
    S(2,1) = RT(2,1);
    S(2,2) = RT(2,2);

    // bottom right 3x3
    S(3,3) = RT(0,0);
    S(3,4) = RT(0,1);
    S(3,5) = RT(0,2);
    S(4,3) = RT(1,0);
    S(4,4) = RT(1,1);
    S(4,5) = RT(1,2);
    S(5,3) = RT(2,0);
    S(5,4) = RT(2,1);
    S(5,5) = RT(2,2);

    return S;
}

rw::math::Q calcDeltaQ(Eigen::Matrix<double, 6, 7> Z, Eigen::Matrix<double, 6, 1> deltaU)
{
    Eigen::Matrix<double, 7, 6> ZInv = LinearAlgebra::pseudoInverse(Z);

    return Q( ZInv*deltaU );
}

rw::math::Q calcDeltaQVision(Eigen::Matrix<double, 8, 7> Z, Eigen::Matrix<double, 8, 1> deltaU)
{
    Eigen::Matrix<double, 7, 8> ZInv = LinearAlgebra::pseudoInverse(Z);

    return Q( ZInv*deltaU );
}

rw::math::Q calcDeltaQSingle(Eigen::Matrix<double, 2, 7> Z, Eigen::Matrix<double, 2, 1> deltaU)
{
    Eigen::Matrix<double, 7, 2> ZInv = LinearAlgebra::pseudoInverse(Z);

    return Q( ZInv*deltaU );
}

double withinVelLimits(const rw::models::Device::Ptr device, rw::math::Q dq, double deltaTime)
{
    rw::math::Q limits = device->getVelocityLimits();

    double exceedingFactor = 1.0;

    for (size_t i = 0; i < dq.size(); i++)
    {
        if (abs(dq[i]/deltaTime) > limits[i])
        {
            double tempFactor = limits[i] / abs(dq[i]/deltaTime);
            cout << "Vel lim factor:" << tempFactor << endl << endl;
            if (tempFactor < exceedingFactor)
            {
                exceedingFactor = tempFactor;
            }
        }
    }

    if (exceedingFactor < 0.005)
        return 0.0;

    return exceedingFactor;
}

rw::math::Q timeScaledQ(rw::math::Q deltaQ, double tauPrime)
{
    /*rw::math::Q newQ();
    for (size_t i = 0; i < deltaQ.size(); ++i)
    {
        newQ[i] = deltaQ[i]*tauPrime;
    }*/

    return deltaQ*tauPrime;;
;
}
