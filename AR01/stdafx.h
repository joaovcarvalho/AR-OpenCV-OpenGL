// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifdef WIN32
	#include <windows.h>    /* includes only in MSWindows not in UNIX,needed by OpenGL in Windows */
	#include <mmsystem.h>
#endif

//Some Windows Headers (For Time, IO, etc.)

// GL 
#include <GL/glew.h>
#include <GL/freeglut.h>

// Utils
#include <iostream>
#include "maths_funcs.h"
#include <cassert>

// Assimp includes

#include <assimp/cimport.h> // C importer
#include <assimp/scene.h> // collects data
#include <assimp/postprocess.h> // various extra operations
#include <stdio.h>
#include <math.h>
#include <vector> // STL dynamic memory.


// OpenCV
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
// TODO: reference additional headers your program requires here
