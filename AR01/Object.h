#pragma once

#include "stdafx.h"
#include "Shader.h"
#include "Texture.h"

class Object
{
protected:
	static int vboId;
	unsigned int vao;
	unsigned int vp_vbo;
	unsigned int vn_vbo;
	unsigned int vt_vbo;

	vec3 position;
	vec3 rotation;
	vec3 scaleVec;

	vec3 rotationAroundDegree;
	vec3 rotationAroundOrigin;

	Shader* shader;
	Texture* texture;

	char* meshName;

	GLuint loc1, loc2, loc3;
	int vertexCounter;
	std::vector<float> g_vp, g_vn, g_vt;

	mat4 model;

	std::vector<Object*> childs;

	void bindVBO();
	GLuint ibo_elements;
	GLuint vbo_vertices_bounding_box;

	vec3 sizeBoundingBox;
	vec3 centerBoundingBox;
public:
	Object(Shader* shader, char* meshName, Texture* texture = NULL);
	Object();

	vec3 minBoundingBox;
	vec3 maxBoundingBox;
	void Object::setVAO(float v);
	void Object::setShader(Shader* s);
	void Object::setVertexCounter(int n);
	void Object::setSizeBoudingBox(vec3);
	void Object::setCenterBoudingBox(vec3);
	void Object::setVertices(std::vector<float>);
	void generateBoundingBox();
	void Object::setBoundingBox(GLuint, GLuint, vec3, vec3, vec3, vec3);
	void Object::setVPVBO(GLuint);

	vec3 getMinVectorInWorld();
	vec3 getMaxVectorInWorld();

	std::vector<float> getVertices();

	void generateObjectBufferMesh(GLuint shaderProgramID, int numVertices);
	bool load_mesh(const char* file_name);

	void display(mat4 view, mat4 projection, mat4 fatherMatrix = identity_mat4());
	void display(cv::Mat mvp);
	void display(mat4 mvp);

	void rotate(float x, float y, float z);
	void rotateAround(float x, float y, float z, vec3 origin);
	void move(float x, float y, float z);
	void scaleAll(float x);

	vec3 getPosition();
	vec3 getRotation();

	void setPosition(vec3);
	void setRotation(vec3);

	void resetState();

	void addChild(Object*);

	void drawBoundingBox();
	void updateBoundingBox();
	virtual Object* clone();

	bool checkCollision(Object*);
	~Object();
protected:

};

