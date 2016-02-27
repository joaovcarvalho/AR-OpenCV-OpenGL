#include "stdafx.h"
#include "Shader.h"
#include "Object.h"
#include "Texture.h"
#include <math.h>
#include <cmath>

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#define NUM_IMAGES_TO_CALIBRATE 15
#define FRAMES_TO_WAIT 5

using namespace std;
using namespace cv;

int width = 800;
int height = 600;

Texture* sunTexture;

Shader* shader;
Shader* nonDiffuseShader;
Shader* nonTextureShader;
Shader* textShader;
Shader* cameraShader;

Object* sun;

// Direction that the camera is towarding
vec3 ViewDir;

vec3 camera_position = vec3(0.0, 0.0, -150.0f);

// Init the view and projection static for every object
int Object::vboId = 0;

mat4 view, projection;

// Framebuffer
GLuint fbo, fbo_texture, rbo_depth;

/* Global */
GLuint program_postproc, attribute_v_coord_postproc, uniform_fbo_texture;
GLuint uniform_fbo_offset;
Shader* postProcessingShader;

GLuint vbo_fbo_vertices;
GLuint mTexture;

VideoCapture cap;

void drawText(float x, float y, std::string text, vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f)) {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, width, 0.0, height);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glUseProgram(textShader->getProgramID());
	glRasterPos2i(x, y);
	glColor4f(color.v[0], color.v[1], color.v[2], color.v[3]);

	for (int i = 0; i<text.size(); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
	}

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

float orientation_rotation_y = 0;

vector< vector<Point2f> > imagePoints(0);
vector< vector<Point3f> > objectPoints(0);
Mat cameraMatrix, distCoeffs;
Mat rvec(3, 1, DataType<double>::type);
Mat tvec(3, 1, DataType<double>::type);
Mat rMat(3, 3, DataType<double>::type);

GLint currProgram;
Mat frame;

bool cameraCalibraded = false;
int wait = 0;

bool cameraCalibration(Mat frame) {
	Size boardSize(9, 6); //interior number of corners
	float squareSize = 1;
	
	Mat gray = frame.clone(); //source image
	cvtColor(frame, gray, CV_BGR2GRAY);
	vector<Point2f> tmp;

	if (wait != 0) {
		wait = wait < 0 ? 0 : wait - 1;
		return false;
	}

	bool patternfound = findChessboardCorners(gray, boardSize, tmp,
		CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
		CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);

	if (!cameraCalibraded) {
		if (patternfound) {
			imagePoints.push_back(tmp);
			objectPoints.push_back(vector<Point3f>());
			cornerSubPix(gray, imagePoints.back(), Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					objectPoints.back().push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));

			drawChessboardCorners(frame, boardSize, Mat(imagePoints.back()), patternfound);
			wait = FRAMES_TO_WAIT;
		}

		if (imagePoints.size() > NUM_IMAGES_TO_CALIBRATE) {
			cameraMatrix = Mat::eye(3, 3, CV_64F);

			distCoeffs = Mat::zeros(8, 1, CV_64F);

			objectPoints.resize(imagePoints.size(), objectPoints[0]);
			
			vector<Mat> rvecs;
			vector<Mat> tvecs;

			//Find intrinsic and extrinsic camera parameters
			double rms = calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix,
				distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

			bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
			cameraCalibraded = true;

			rvec = rvecs.back();
			tvec = tvecs.back();
			Rodrigues(rvec, rMat);
			cout << "Camera Calibraded !!" << endl;
		}
	}
	else {
		if (patternfound) {
			cornerSubPix(gray, tmp, Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			
			vector<Point3f> currentObjectsPoints; 
			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					currentObjectsPoints.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
			
			solvePnP(currentObjectsPoints, tmp, cameraMatrix, distCoeffs, rvec, tvec);

			Rodrigues(rvec, rMat);
			
			drawChessboardCorners(frame, boardSize, Mat(tmp), patternfound);
			return true;
		}
	}

	return false;
}

mat4 generateFrustrumWithCamera(float nearf, float farf) {
	float imageWidth = width;
	float imageHeight = height;

	Mat expandedR;
	Rodrigues(rvec, expandedR);

	Mat Rt = Mat::zeros(4, 4, CV_64FC1);
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			Rt.at<double>(y, x) = expandedR.at<double>(y, x);
		}
	}

	Rt.at<double>(0, 3) = -tvec.at<double>(0, 0) + 4;
	Rt.at<double>(1, 3) = -tvec.at<double>(1, 0) - 1;
	Rt.at<double>(2, 3) = tvec.at<double>(2, 0) + 4;

	Rt.at<double>(3, 3) = 1.0;

	Mat invertAxis = Mat::eye(Size(4,4), CV_64FC1 );
	invertAxis.at<double>(2, 2) = -1;
	invertAxis.at<double>(1, 1) = -1;
	invertAxis.at<double>(0, 0) = -1;

	Rt = invertAxis * Rt;

	float left, right, top, bottom;
	right = (width - cameraMatrix.at<double>(0, 2))*nearf/ (cameraMatrix.at<double>(0,0) );
	left = -cameraMatrix.at<double>(0, 2)*nearf / (cameraMatrix.at<double>(0, 0));
	top = (height - cameraMatrix.at<double>(1, 2))*nearf / (cameraMatrix.at<double>(1, 1));
	bottom = (-cameraMatrix.at<double>(1, 2))*nearf / (cameraMatrix.at<double>(1, 1));

	Mat projMat = Mat::zeros(4, 4, CV_64FC1);
	projMat.at<double>(0, 0) = 2 * nearf / (right - left);
	projMat.at<double>(0, 2) = (right + left) / (right - left);
	projMat.at<double>(1, 1) = 2 * nearf / (top - bottom);
	projMat.at<double>(1, 2) = (top + bottom) / (top - bottom);
	projMat.at<double>(2, 2) = -(farf + nearf) / (farf - nearf);
	projMat.at<double>(2, 3) = -2 * farf*nearf / (farf - nearf);
	projMat.at<double>(3, 2) = -1;

	mat4 result = identity_mat4();
	Mat mvp = projMat * Rt;

	for (int i = 0; i < mvp.rows; i++)
	{
		for (int j = 0; j < mvp.cols; j++)
		{
			result.m[j * 4 + i] = (float)mvp.at<double>(i, j);
		}
	}

	return result;
}

void display() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// tell GL to only draw onto a pixel if the shape is closer to the viewer
	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDepthMask(GL_FALSE);
	cap >> frame;
	bool found = false;
	if (!frame.empty()) {
		// Flips the frame as the opencv give us the image inverted
		flip(frame, frame, -1);

		// Camera Calibration
		found = cameraCalibration(frame);
		
		glUseProgram(postProcessingShader->getProgramID());
		// Bind the texture to the image
		glBindTexture(GL_TEXTURE_2D, mTexture);
		glEnable(GL_TEXTURE_2D);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);

		// Draw a Polygon with the texture
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(1.0f, 1.0f, 1000.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 1000.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 1000.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(1.0f, -1.0f, 1000.0f);
		glEnd();

		glFlush();

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	else {
		std::cout << "Error getting frame from camera." << endl;
		return;
	}

	glDepthMask(GL_TRUE);
	
	stringstream text;
	if (!cameraCalibraded) {
		text << "Calibrating Camera: " << imagePoints.size() << "/" << NUM_IMAGES_TO_CALIBRATE;
	}
	else {
		text << "Camera Calibrated";
	}
	
	drawText(20, 20, text.str());


	if (cameraCalibraded && found) {
		mat4 mvp = generateFrustrumWithCamera(0.1, 5000);
		sun->rotate(0, 1.0f, 0);
		sun->display(mvp);
	}

	glutSwapBuffers();
}

void init()
{
	cout << "Initialization of assets" << endl;

	// Set up the shaders
	shader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/simpleFragmentShader.hlsl");
	nonDiffuseShader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/nonDiffuseFragmentShader.hlsl");
	nonTextureShader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/nonTextureFragmentShader.hlsl");

	textShader = new Shader("Shaders/identityVertexShader.hlsl", "Shaders/identityFragmentShader.hlsl");

	postProcessingShader = new Shader("Shaders/postprocessingVertex.hlsl", "Shaders/postprocessingFragment.hlsl");
	cout << "Shaders finished loading." << endl;

	// Instantiate the textures
	sunTexture = new Texture("models/sun.jpg");

	//initializating the texture mapping
	glGenTextures(1, &mTexture);
	glBindTexture(GL_TEXTURE_2D, mTexture);
	glEnable(GL_TEXTURE_2D);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	cout << "Textures finished loading." << endl;

	// Instantitate the objects
	sun = new Object(nonDiffuseShader, "models/Earth.obj", sunTexture);
	cout << "Objects finished loading." << endl;

	// Doing initial transformations
	//sun->move(5.0,10.0, 0.0);
	sun->scaleAll(0.025);

	cout << "Finished initialization" << endl;
}

void updateScene() {
	// Draw the next frame
	glutPostRedisplay();
}

void exit() {
	glDeleteRenderbuffers(1, &rbo_depth);
	glDeleteTextures(1, &fbo_texture);
	glDeleteFramebuffers(1, &fbo);

	/* free_resources */
	glDeleteBuffers(1, &vbo_fbo_vertices);
}

int main(int argc, char** argv) {
	// Set up the window
	cap = VideoCapture(0);
	cap.set(CAP_PROP_FPS, 60.0);
	cap >> frame;
	width  = frame.size().width;
	height = frame.size().height;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(frame.size().width, frame.size().height);
	glutCreateWindow("Space Jam");

	// Tell glut where the display function is
	glutDisplayFunc(display);
	glutIdleFunc(updateScene);

	// A call to glewInit() must be done after glut is initialized!
	GLenum res = glewInit();
	// Check for any errors
	if (res != GLEW_OK) {
		fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
		return 1;
	}
	// Set up your objects and shaders
	init();
	// Begin infinite event loop
	glutMainLoop();

	atexit(exit);
	return 0;
}


