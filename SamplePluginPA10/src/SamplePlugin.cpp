#include "SamplePlugin.hpp"
#include "functions.h"

#include <rws/RobWorkStudio.hpp>

#include <QPushButton>

#include <rw/loaders/ImageLoader.hpp>
#include <rw/loaders/WorkCellFactory.hpp>

using namespace rw::loaders;

#include <functional>
#include <iostream>
#include "vision_color_marker.hpp"

using namespace rws;

using namespace cv;

using namespace std::placeholders;

SamplePlugin::SamplePlugin():
    RobWorkStudioPlugin("SamplePluginUI", QIcon(":/pa_icon.png"))
{
	setupUi(this);

	_timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timer()));

	// now connect stuff from the ui component
	connect(_btn0    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
	connect(_btn1    ,SIGNAL(pressed()), this, SLOT(btnPressed()) );
    //connect(_spinBox  ,SIGNAL(valueChanged(int)), this, SLOT(btnPressed()) );
    connect(_markerBox, QOverload<const QString &>::of(&QComboBox::activated),
            [=](const QString &text){ if (text == "Marker1"){_markerPath = "/home/quist/Documents/7-Semester/Rovi_Project/SamplePluginPA10/markers/Marker1.ppm";}
        else if (text == "Marker2a"){_markerPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/markers/Marker2a.ppm";}
        else if (text == "Marker2b"){_markerPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/markers/Marker2b.ppm";}
        else if (text == "Marker4"){_markerPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/markers/Marker3.ppm";}
        Image::Ptr image;
        image = ImageLoader::Factory::load(_markerPath);
        _textureRender->setImage(*image);
        getRobWorkStudio()->updateAndRepaint();}
    );
    connect(_backgroundBox, QOverload<const QString &>::of(&QComboBox::activated),
            [=](const QString &text){ string path;
        if (text == "Color1"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/color1.ppm";}
        else if (text == "Color2"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/color2.ppm";}
        else if (text == "Color3"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/color3.ppm";}
        else if (text == "Lines1"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/lines1.ppm";}
        else if (text == "Texture1"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/texture1.ppm";}
        else if (text == "Texture2"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/texture2.ppm";}
        else if (text == "Texture3"){path = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/backgrounds/texture3.ppm";}
        Image::Ptr image;
        image = ImageLoader::Factory::load(path);
        _bgRender->setImage(*image);
        getRobWorkStudio()->updateAndRepaint();}
    );
    connect(_trackingType, QOverload<const QString &>::of(&QComboBox::activated),
            [=](const QString &text){
        if (text == "Three Markers"){_trackType = "3markers";}
        else if (text == "Single Marker"){_trackType = "1marker";}
        else if (text == "Vision"){_trackType = "vis";}}
    );
    connect(_motionSpeed, QOverload<const QString &>::of(&QComboBox::activated),
            [=](const QString &text){
        if (text == "Slow Marker Motion"){_motionPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/motions/MarkerMotionSlow.txt";}
        else if (text == "Medium Marker Motion"){_motionPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/motions/MarkerMotionMedium.txt";}
        else if (text == "Fast Marker Motion"){_motionPath = "/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/motions/MarkerMotionFast.txt";}}
    );
    connect(_spinBox  ,SIGNAL(valueChanged(int)), this, SLOT(btnPressed()) );

	Image textureImage(300,300,Image::GRAY,Image::Depth8U);
	_textureRender = new RenderImage(textureImage);
	Image bgImage(0,0,Image::GRAY,Image::Depth8U);
	_bgRender = new RenderImage(bgImage,2.5/1000.0);
	_framegrabber = NULL;
}

SamplePlugin::~SamplePlugin()
{
    delete _textureRender;
    delete _bgRender;
}

void SamplePlugin::initialize() {
	log().info() << "INITALIZE" << "\n";

	getRobWorkStudio()->stateChangedEvent().add(std::bind(&SamplePlugin::stateChangedListener, this, _1), this);

	// Auto load workcell
    WorkCell::Ptr wc = WorkCellLoader::Factory::load("/home/quist/Documents/7-Semester/Rovi_Project/final_project/PA10WorkCell/ScenePA10RoVi1.wc.xml");
	getRobWorkStudio()->setWorkCell(wc);


	// Load Lena image
	Mat im, image;
    im = imread("/home/quist/Documents/7-Semester/Rovi_Project/final_project/SamplePluginPA10/src/lena.bmp", CV_LOAD_IMAGE_COLOR); // Read the file
	cvtColor(im, image, CV_BGR2RGB); // Switch the red and blue color channels
	if(! image.data ) {
		RW_THROW("Could not open or find the image: please modify the file path in the source code!");
	}
	QImage img(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888); // Create QImage from the OpenCV image
    _label->setPixmap(QPixmap::fromImage(img)); // Show the image at the label in the plugin

}

void SamplePlugin::open(WorkCell* workcell)
{
    log().info() << "OPEN" << "\n";
    _wc = workcell;
    _state = _wc->getDefaultState();

    log().info() << workcell->getFilename() << "\n";

    if (_wc != NULL)
    {
        // Add the texture render to this workcell if there is a frame for texture
        Frame* textureFrame = _wc->findFrame("MarkerTexture");
        if (textureFrame != NULL) {
            getRobWorkStudio()->getWorkCellScene()->addRender("TextureImage",_textureRender,textureFrame);
        }
        // Add the background render to this workcell if there is a frame for texture
        Frame* bgFrame = _wc->findFrame("Background");
        if (bgFrame != NULL) {
            getRobWorkStudio()->getWorkCellScene()->addRender("BackgroundImage",_bgRender,bgFrame);
        }

        // Create a GLFrameGrabber if there is a camera frame with a Camera property set
        Frame* cameraFrame = _wc->findFrame("CameraSim");
        if (cameraFrame != NULL) {
            if (cameraFrame->getPropertyMap().has("Camera")) {
                // Read the dimensions and field of view
                double fovy;
                int width,height;
                std::string camParam = cameraFrame->getPropertyMap().get<std::string>("Camera");
                std::istringstream iss (camParam, std::istringstream::in);
                iss >> fovy >> width >> height;
                // Create a frame grabber
                _framegrabber = new GLFrameGrabber(width,height,fovy);
                SceneViewer::Ptr gldrawer = getRobWorkStudio()->getView()->getSceneViewer();
                _framegrabber->init(gldrawer);
            }
        }
        // Get device and check if it has been loaded correctly
        _device = _wc->findDevice(_device_name);
        if(_device == nullptr)
            RW_THROW("Device " << _device_name << " was not found!");
    }
}


void SamplePlugin::close() {
    log().info() << "CLOSE" << "\n";

    // Stop the timer
    _timer->stop();
    // Remove the texture render
	Frame* textureFrame = _wc->findFrame("MarkerTexture");
	if (textureFrame != NULL) {
		getRobWorkStudio()->getWorkCellScene()->removeDrawable("TextureImage",textureFrame);
	}
	// Remove the background render
	Frame* bgFrame = _wc->findFrame("Background");
	if (bgFrame != NULL) {
		getRobWorkStudio()->getWorkCellScene()->removeDrawable("BackgroundImage",bgFrame);
	}
	// Delete the old framegrabber
	if (_framegrabber != NULL) {
		delete _framegrabber;
	}
	_framegrabber = NULL;
	_wc = NULL;
}

Mat SamplePlugin::toOpenCVImage(const Image& img) {
    Mat res(img.getHeight(),img.getWidth(), CV_8UC3);
	res.data = (uchar*)img.getImageData();
	return res;
}


void SamplePlugin::btnPressed() {
    QObject *obj = sender();
	if(obj==_btn0){
        log().info() << "Button 0\n";

	} else if(obj==_btn1){
		log().info() << "Button 1\n";
		// Toggle the timer on and off
		if (!_timer->isActive())
        {
            _markerTransforms = loadMarkers(_motionPath);
            _markerIndex = 0;

            Frame* camFrame = _wc->findFrame("Camera");
            MovableFrame* markerFrame = (MovableFrame*)_wc->findFrame("Marker");
            markerFrame->setTransform(_markerTransforms[_markerIndex], _state);

            Q start(7, 0, -0.65, 0, 1.8, 0, 0.42, 0);
            _device->setQ(start, _state);
            getRobWorkStudio()->setState(_state);

            if (_trackType == "3markers")
            {   _tracker = "three markers";
                vector<Vector3D<double>> markerPoints;
                markerPoints.emplace_back(Vector3D<double>(0.15, 0.15, 0));
                markerPoints.emplace_back(Vector3D<double>(-0.15, 0.15, 0));
                markerPoints.emplace_back(Vector3D<double>(0.15, -0.15, 0));
                _targetUV = calcUV(markerFrame, camFrame, markerPoints, _state, _focalLength, _z);
            }
            else if (_trackType == "vis")
            {   _tracker = "vision";
                Frame* cameraFrame = _wc->findFrame("CameraSim");
                _framegrabber->grab(cameraFrame, _state);
                const Image& image = _framegrabber->getImage();
                // Convert to OpenCV image
                Mat im = toOpenCVImage(image);
                Mat imflip;
                cv::flip(im, imflip, 0);

                vector<Point2f> markerPoints = findColorMarkerPoints(im);
                _targetUV = calcUVVision(markerPoints, im);
            }
            else if (_trackType == "1marker")
            {   _tracker = "single marker";
                vector<Vector3D<double>> markerPoints;
                markerPoints.emplace_back(Vector3D<double>(0, 0, 0));
                _targetUV = calcUV(markerFrame, camFrame, markerPoints, _state, _focalLength, _z);
            }

            if (!_timer->isActive())
                _timer->start(100); // run 10 Hz
            else
            {
                _timer->stop();
                _markerIndex = 0;

                Frame* camFrame = _wc->findFrame("Camera");
                MovableFrame* markerFrame = (MovableFrame*)_wc->findFrame("Marker");
                markerFrame->setTransform(_markerTransforms[_markerIndex], _state);

                Q start(7, 0, -0.65, 0, 1.8, 0, 0.42, 0);
                _device->setQ(start, _state);
                getRobWorkStudio()->setState(_state);
                _targetUV.clear();
            }
        }
		else
			_timer->stop();
    }else if(obj==_spinBox){
        log().info() << "spin value:" << _spinBox->value() << "\n";
        _delta_time = _spinBox->value();
    }
}

void SamplePlugin::timer() {
	if (_framegrabber != NULL) {
        if (_markerIndex >= _markerTransforms.size())
        {
            _timer->stop();
            return;
        }


        // Move the marker!
        auto markerTransform = _markerTransforms[_markerIndex];
        _markerIndex++;
        MovableFrame* markerFrame = (MovableFrame*)_wc->findFrame("Marker");
        markerFrame->setTransform(markerTransform, _state);

        Frame* camFrame = _wc->findFrame("Camera");

        Jacobian J = _device->baseJframe(camFrame, _state);
        rw::math::Q deltaQ;

        // Get the image as a RW image
        Frame* cameraFrame = _wc->findFrame("CameraSim");
        _framegrabber->grab(cameraFrame, _state);
        const Image& image = _framegrabber->getImage();

        // Convert to OpenCV image
        Mat im = toOpenCVImage(image);
        Mat imflip;
        cv::flip(im, imflip, 0);

        // Show in QLabel
        QImage img(imflip.data, imflip.cols, imflip.rows, imflip.step, QImage::Format_RGB888);
        QPixmap p = QPixmap::fromImage(img);
        unsigned int maxW = 400;
        unsigned int maxH = 800;
        _label->setPixmap(p.scaled(maxW,maxH,Qt::KeepAspectRatio));

        // Following hardcoded path with 3 markers
        if (_tracker == "three markers")
        {
            vector<Vector3D<double>> markerPoints;
            markerPoints.emplace_back(Vector3D<double>(0.15, 0.15, 0));
            markerPoints.emplace_back(Vector3D<double>(-0.15, 0.15, 0));
            markerPoints.emplace_back(Vector3D<double>(0.15, -0.15, 0));
            vector<double> uv = calcUV(markerFrame, camFrame, markerPoints, _state, _focalLength, _z);

            Eigen::Matrix<double, 6, 1> deltaU;
            for (int i = 0; i < 6; i++)
                deltaU(i,0) = _targetUV[i]-uv[i];

            Eigen::Matrix<double, 6, 6> J_img = calcImgJ(uv, _z, _focalLength);

            Eigen::Matrix<double, 6, 6> Sq = calcS(_device, _state, camFrame);

            Eigen::Matrix<double, 6, 7> Z_img = J_img*Sq*J.e();

            deltaQ = calcDeltaQ(Z_img, deltaU);

        }// Following Vision tracked markers
        else if (_tracker == "vision" )
        {
            vector<Point2f> markerPoints = findColorMarkerPoints(im);

            vector<double> uv = calcUVVision(markerPoints, im);

            Eigen::Matrix<double, 8, 1> deltaU;
            for (int i = 0; i < 8; i++)
                deltaU(i,0) = _targetUV[i]-uv[i];

            Eigen::Matrix<double, 8, 6> J_img = calcImgJVision(uv, _z, _focalLength);

            Eigen::Matrix<double, 6, 6> Sq = calcS(_device, _state, camFrame);

            Eigen::Matrix<double, 8, 7> Z_img = J_img*Sq*J.e();

            deltaQ = calcDeltaQVision(Z_img, deltaU);

        }// Following a single marker
        else if (_tracker == "single marker")
        {
            vector<Vector3D<double>> markerPoints;
            markerPoints.emplace_back(Vector3D<double>(0, 0, 0));
            vector<double> uv = calcUV(markerFrame, camFrame, markerPoints, _state, _focalLength, _z);

            Eigen::Matrix<double, 2, 1> deltaU;
            for (int i = 0; i < 2; i++)
                deltaU(i,0) = _targetU1-uv[i];

            Eigen::Matrix<double, 2, 6> J_img = calcImgJSingle(uv, _z, _focalLength);

            Eigen::Matrix<double, 6, 6> Sq = calcS(_device, _state, camFrame);

            Eigen::Matrix<double, 2, 7> Z_img = J_img*Sq*J.e();

            deltaQ = calcDeltaQSingle(Z_img, deltaU);

        }

        double velLim = withinVelLimits(_device, deltaQ, _delta_time);

        if (velLim == 1.0)
		{
            rw::math::Q newQ = _device->getQ(_state)+deltaQ;
            _device->setQ(newQ, _state);
		}
		else
		{
            rw::math::Q newQ = _device->getQ(_state)+timeScaledQ(deltaQ, velLim);
            _device->setQ(newQ, _state);
        }


        getRobWorkStudio()->setState(_state);
	}
}

void SamplePlugin::stateChangedListener(const State& state) {
  _state = state;
}
