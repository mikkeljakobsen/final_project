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

tuple<double, double> calcUV(MovableFrame* markerFrame, Frame* cameraFrame, Vector3D<double> posInMarkerFrame, State state, double focalLength, double z)
{
    rw::math::Vector3D<double> markerPosInCamFrame = cameraFrame->fTf(markerFrame, state)*posInMarkerFrame;
    double u = (focalLength*markerPosInCamFrame[0])/z;
    double v = (focalLength*markerPosInCamFrame[1])/z;

    return {u, v};
}


Eigen::Matrix<double, 6, 6> calcImgJ(vector<double> uv, double z, double f)
{
    static const int numberOfMarkers = 3;//uv.size();
    static constexpr int rows = 2*numberOfMarkers;
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

double withinVelLimits(const rw::models::Device::Ptr device, rw::math::Q dq, double deltaTime)
{
    rw::math::Q limits = device->getVelocityLimits();

    double exceedingFactor = 1.0;

    for (size_t i = 0; i < dq.size(); i++)
    {
        if (abs(dq[i]/deltaTime) > limits[i])
        {
            cout << "Limit of joint " << i << " is " << limits[i] << endl;
            double tempFactor = limits[i] / abs(dq[i]/deltaTime);
            cout << "Vel lim factor:" << tempFactor << endl << endl;
            if (tempFactor < exceedingFactor)
            {
                exceedingFactor = tempFactor;
            }
        }
    }

    return exceedingFactor;
}

rw::math::Q timeScaledQ(rw::math::Q deltaQ, double tauPrime)
{
    rw::math::Q newQ;

    for (size_t i = 0; i < deltaQ.size(); ++i)
    {
        newQ[i] = deltaQ[i]*tauPrime;
    }

    return newQ;
}









/*
// This function calculates delta U as in Equation 4.13. The output class is a velocity screw as that is a 6D vector with a positional and rotational part
// What a velocity screw really is is not important to this class. For our purposes it is only a container.
rw::math::VelocityScrew6D<double> calculateDeltaU(const rw::math::Transform3D<double>& baseTtool, const rw::math::Transform3D<double>& baseTtool_desired) {
    // Calculate the positional difference, dp
    rw::math::Vector3D<double> dp = baseTtool_desired.P() - baseTtool.P();

    // Calculate the rotational difference, dw
    rw::math::EAA<double> dw(baseTtool_desired.R() * rw::math::inverse(baseTtool.R()));

    return rw::math::VelocityScrew6D<double>(dp, dw);
}

// The inverse kinematics algorithm needs to know about the device, the tool frame and the desired pose. These parameters are const since they are not changed by inverse kinematics
// We pass the state and the configuration, q, as value so we have copies that we can change as we want during the inverse kinematics.
rw::math::Q algorithm1(const rw::models::Device::Ptr device, rw::kinematics::State state, const rw::kinematics::Frame* tool, const rw::math::Transform3D<double> baseTtool_desired, rw::math::Q q)
{
    // We need an initial base to tool transform and the positional error at the start (deltaU)
    rw::math::Transform3D<> baseTtool = device->baseTframe(tool, state);
    rw::math::VelocityScrew6D<double> deltaU = calculateDeltaU(baseTtool, baseTtool_desired);

    // Epsilon is the desired tolerance on the final position.
    const double epsilon = 0.0001;

    while(deltaU.norm2() > epsilon) {
        rw::math::Jacobian J = device->baseJframe(tool, state);

        // Because this is NOT a 6 DOF robot but rather a 7 DOF we use the pseudoInverse.
        rw::math::Jacobian Jinv = LinearAlgebra::pseudoInverse(J.e());

        rw::math::Q deltaQ = Jinv*deltaU.e();

        // Here we add the change in configuration to the current configuration and move the robot to that position.
        q += deltaQ;
        device->setQ(q, state);

        // We need to calculate the forward dynamics again since the robot has been moved
        baseTtool = device->baseTframe(tool, state); // This line performs the forward kinematics (Programming Exercise 3.4)

        // Update the cartesian position error
        deltaU = calculateDeltaU(baseTtool, baseTtool_desired);
    }
    return q;
}*/
