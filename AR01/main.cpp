#include "stdafx.h"
#include "Shader.h"
#include "Object.h"
#include "Texture.h"
#include <math.h>
#include <cmath>

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace std;
using namespace cv;

int width = 800;
int height = 600;

Texture* earthTexture;
Texture* playerTexture;
Texture* moonTexture;
Texture* sunTexture;

Shader* shader;
Shader* nonDiffuseShader;
Shader* nonTextureShader;
Shader* skyBoxShader;
Shader* textShader;
Shader* cameraShader;

Object* earth;
Object* moon;
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

mat4 updateCamera() {
	// Camera rotation control
	mat4 view = translate(identity_mat4(), vec3(0, 0, -500.0));

	if (cameraCalibraded) {
		view = identity_mat4();
		for (int i = 0; i < rMat.rows; i++)
		{
			for (int j = 0; j < rMat.cols; j++)
			{
				view.m[j * 4 + i] = rMat.at<double>(i, j);
			}
		}
		
		view.m[12] =  tvec.at<double>(0, 0);
		view.m[13] =  tvec.at<double>(1, 0);
		view.m[14] =  tvec.at<double>(2, 0);
	}

	mat4 convertOpenCV2OpenGL = identity_mat4();
	convertOpenCV2OpenGL.m[5] = -1;
	convertOpenCV2OpenGL.m[10] = -1;

	return convertOpenCV2OpenGL * view;
}

bool cameraCalibration(Mat frame) {
	Size boardSize(9, 6); //interior number of corners
	float squareSize = 5;
	Mat gray = frame.clone(); //source image
	cvtColor(frame, gray, CV_BGR2GRAY);
	vector<Point2f> tmp;

	bool patternfound = findChessboardCorners(gray, boardSize, tmp,
		CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
		+ CALIB_CB_FAST_CHECK);

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
		}

		if (imagePoints.size() > 15) {
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
	Rt.at<double>(0, 3) = tvec.at<double>(0, 0);
	Rt.at<double>(1, 3) = tvec.at<double>(1, 0);
	Rt.at<double>(2, 3) = tvec.at<double>(2, 0);

	Rt.at<double>(3, 3) = 1.0;

	//OpenGL has reversed Y & Z coords
	Mat reverseYZ = Mat::eye(4, 4, CV_64FC1);
	reverseYZ.at<double>(1, 1) = reverseYZ.at<double>(0,0) = reverseYZ.at<double>(2, 2) = -1;

	//since we are in landscape mode
	Mat rot2D = Mat::eye(4, 4, CV_64FC1);
	rot2D.at<double>(0, 0) = rot2D.at<double>(1, 1) = 0;
	rot2D.at<double>(0, 1) = 1;
	rot2D.at<double>(1, 0) = -1;

	Mat projMat = Mat::zeros(4, 4, CV_64FC1);
	projMat.at<double>(0, 0) = 2 * cameraMatrix.at<double>(0, 0) / imageWidth;
	projMat.at<double>(0, 2) = -1 + (2 * cameraMatrix.at<double>(0, 2) / imageWidth);
	projMat.at<double>(1, 1) = 2 * cameraMatrix.at<double>(1, 1) / imageHeight;
	projMat.at<double>(1, 2) = -1 + (2 * cameraMatrix.at<double>(1, 2) / imageHeight);
	projMat.at<double>(2, 2) = -(farf + nearf) / (farf - nearf);
	projMat.at<double>(2, 3) = -2 * farf*nearf / (farf - nearf);
	projMat.at<double>(3, 2) = -1;

	Mat mvMat = reverseYZ * Rt;
	//projMat = rot2D * projMat;

	mat4 result = identity_mat4();
	Mat mvp = projMat * mvMat;

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

	glGetIntegerv(GL_CURRENT_PROGRAM, &currProgram);

	glDepthMask(GL_FALSE);
	cap >> frame;
	bool found = false;
	if (!frame.empty()) {
		// Camera Calibration
		found = cameraCalibration(frame);

		// Flips the frame as the opencv give us the image inverted
		flip(frame, frame, -1);
		// Bind the texture to the image
		glBindTexture(GL_TEXTURE_2D, mTexture);
		glEnable(GL_TEXTURE_2D);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);

		//glUseProgram(cameraShader->getProgramID());
		// Draw a Polygon with the texture
		glBegin(GL_POLYGON);
		glTexCoord2f(0.0f, 0.0f); glVertex3f(1.0f, 1.0f, 1000.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 1000.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 1000.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex3f(1.0f, -1.0f, 1000.0f);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	else {
		std::cout << "Error getting frame from camera." << endl;
		return;
	}

	glDepthMask(GL_TRUE);
	
	if (cameraCalibraded && found) {
		mat4 mvp = generateFrustrumWithCamera(5, 1000);
		//moon->rotate(0, 0.6f, 0);
		//earth->rotate(0, 0.5f, 0);
		//sun->rotate(0, 1.0f, 0);
		sun->display(mvp);
	}

	glUseProgram(currProgram);
	
	glutSwapBuffers();
}



void initCamera() {
	cap = VideoCapture(0);
	cap.set(CAP_PROP_FPS, 60.0);
}


void init()
{
	cout << "Initialization of assets" << endl;

	// Set up the shaders
	shader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/simpleFragmentShader.hlsl");
	nonDiffuseShader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/nonDiffuseFragmentShader.hlsl");
	nonTextureShader = new Shader("Shaders/simpleVertexShader.hlsl", "Shaders/nonTextureFragmentShader.hlsl");

	textShader = new Shader("Shaders/identityVertexShader.hlsl", "Shaders/identityFragmentShader.hlsl");
	skyBoxShader = new Shader("Shaders/skyBoxVertexShader.hlsl", "Shaders/skyBoxFragmentShader.hlsl");

	cameraShader = new Shader("Shaders/cameraVertexShader.hlsl", "Shaders/cameraFragmentShader.hlsl");
	postProcessingShader = new Shader("Shaders/postprocessingVertex.hlsl", "Shaders/postprocessingFragment.hlsl");
	cout << "Shaders finished loading." << endl;

	// Instantiate the textures
	earthTexture = new Texture("models/Earth_Diffuse.jpg");
	moonTexture = new Texture("models/moonmap2k.jpg");
	sunTexture = new Texture("models/sun.jpg");
	playerTexture = new Texture("models/Maps/zqw1b.jpg");

	//initializating the texture mapping
	glGenTextures(1, &mTexture);
	glBindTexture(GL_TEXTURE_2D, mTexture);
	glEnable(GL_TEXTURE_2D);
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	cout << "Textures finished loading." << endl;

	// Instantitate the objects
	earth = new Object(nonTextureShader, "models/Earth.obj", earthTexture);
	moon = new Object(nonTextureShader, "models/Earth.obj", moonTexture);
	sun = new Object(nonTextureShader, "models/Earth.obj", sunTexture);
	cout << "Objects finished loading." << endl;

	// Doing initial transformations
	//sun->scaleAll(10);
	earth->rotate(0.0, 0.0, 180.0);
	earth->move(50.0, 0.0, 100.0);
	earth->scaleAll(0.5);

	moon->scaleAll(0.1);
	moon->move(75.0f, 0, 0.0f);

	earth->addChild(moon);
	sun->addChild(earth);

	cout << "Finished initialization" << endl;
	
	initCamera();

	/* init_resources */
	/* Create back-buffer, used for post-processing */

	// Post processing config
	GLchar* attribute_name = "v_coord";
	attribute_v_coord_postproc = glGetAttribLocation(postProcessingShader->getProgramID(), attribute_name);
	if (attribute_v_coord_postproc == -1) {
		fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
	}

	GLchar* uniform_name = "fbo_texture";
	uniform_fbo_texture = glGetUniformLocation(postProcessingShader->getProgramID(), uniform_name);
	if (uniform_fbo_texture == -1) {
		fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
	}

	/* init_resources */
	GLfloat fbo_vertices[] = {
		-1, -1,
		1, -1,
		-1,  1,
		1,  1,
	};
	glGenBuffers(1, &vbo_fbo_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_fbo_vertices);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fbo_vertices), fbo_vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/* Texture */
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &fbo_texture);
	glBindTexture(GL_TEXTURE_2D, fbo_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	/* Depth buffer */
	glGenRenderbuffers(1, &rbo_depth);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	/* Framebuffer to link everything together */
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);
	GLenum status;
	if ((status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "glCheckFramebufferStatus: error %p", status);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(width, height);
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


