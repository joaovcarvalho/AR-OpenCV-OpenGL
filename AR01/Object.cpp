#include "Object.h"
#include "stdafx.h"
#include <cassert>

Object::Object(Shader* shader, char* meshName, Texture* texture)
{
	this->texture = texture;
	this->meshName = meshName;
	this->vao = NULL;
	this->vertexCounter = 0;
	this->vp_vbo = 0;
	this->vn_vbo = 0;
	this->vt_vbo = 0;

	load_mesh(meshName);
	generateObjectBufferMesh(shader->getProgramID(), this->vertexCounter);

	this->shader = shader;

	this->model = identity_mat4();
	this->position = vec3(0, 0, 0);
	this->rotation = vec3(0, 0, 0);
	this->scaleVec = vec3(1,1,1);


	this->childs = std::vector<Object*>();

	//generateBoundingBox();
}

Object::Object() {
	this->model = identity_mat4();
	this->position = vec3(0, 0, 0);
	this->rotation = vec3(0, 0, 0);
	this->scaleVec = vec3(1, 1, 1);

	this->childs = std::vector<Object*>();
}


Object::~Object()
{
	glDeleteBuffers(1, &this->vao);	
}

vec3 Object::getPosition() {
	return this->position;
}

void Object::setPosition(vec3 v) {
	this->position = v;
}

void Object::setRotation(vec3 v) {
	this->rotation = v;
}


vec3 Object::getRotation() {
	return this->rotation;
}

void Object::rotate(float x, float y, float z) {
	this->rotation += vec3(x,y,z);
	//updateBoundingBox();
}

void Object::scaleAll(float i) {
	this->scaleVec = vec3(i, i, i);
}

void Object::rotateAround(float x, float y, float z, vec3 origin) {
	this->rotationAroundDegree += vec3(x, y, z);
	this->rotationAroundOrigin = origin;
}

void Object::move(float x, float y, float z) {
	this->position += vec3(x, y, z);
}

void Object::resetState() {
	this->model = identity_mat4();
	position = vec3(0, 0, 0);
	rotation = vec3(0, 0, 0);
}

void Object::addChild(Object* c) {
	this->childs.push_back(c);
}



void Object::bindVBO() {
	GLuint shaderProgramID = this->shader->getProgramID();
	loc1 = glGetAttribLocation(shaderProgramID, "vertex_position");
	loc2 = glGetAttribLocation(shaderProgramID, "vertex_normal");
	loc3 = glGetAttribLocation(shaderProgramID, "vertex_texture");
	
	if (this->vao == NULL) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// Vertexs binding
		glEnableVertexAttribArray(loc1);
		glBindBuffer(GL_ARRAY_BUFFER, vp_vbo);
		glVertexAttribPointer(loc1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		// Normal bindings
		glEnableVertexAttribArray(loc2);
		glBindBuffer(GL_ARRAY_BUFFER, vn_vbo);
		glVertexAttribPointer(loc2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		if (this->texture != NULL) {
			glEnableVertexAttribArray(loc3);
			glBindBuffer(GL_ARRAY_BUFFER, vt_vbo);
			glVertexAttribPointer(loc3, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		}
	}else{
		glBindVertexArray(vao);
	}
}

bool Object::load_mesh(const char* file_name) {
	const aiScene* scene = aiImportFile(file_name, aiProcess_Triangulate);

	if (!scene) {
		fprintf(stderr, "ERROR: reading mesh %s\n", file_name);
		return false;
	}

	/*
	printf("  %i animations\n", scene->mNumAnimations);
	printf("  %i cameras\n", scene->mNumCameras);
	printf("  %i lights\n", scene->mNumLights);
	printf("  %i materials\n", scene->mNumMaterials);
	printf("  %i meshes\n", scene->mNumMeshes);
	printf("  %i textures\n", scene->mNumTextures);
	*/

	if (scene->HasMaterials())
	{
		for (unsigned int i = 0; i < scene->mNumMaterials; i++)
		{
			const aiMaterial* material = scene->mMaterials[i];

			if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
				aiString Path;
			}
		}
	}



	for (unsigned int m_i = 0; m_i < scene->mNumMeshes; m_i++) {
		const aiMesh* mesh = scene->mMeshes[m_i];
		//printf("    %i vertices in mesh\n", mesh->mNumVertices);
		this->vertexCounter = mesh->mNumVertices;

		for (unsigned int v_i = 0; v_i < mesh->mNumVertices; v_i++) {
			if (mesh->HasPositions()) {
				const aiVector3D* vp = &(mesh->mVertices[v_i]);
				//printf ("      vp %i (%f,%f,%f)\n", v_i, vp->x, vp->y, vp->z);
				g_vp.push_back(vp->x);
				g_vp.push_back(vp->y);
				g_vp.push_back(vp->z);
			}
			if (mesh->HasNormals()) {
				const aiVector3D* vn = &(mesh->mNormals[v_i]);
				//printf ("      vn %i (%f,%f,%f)\n", v_i, vn->x, vn->y, vn->z);
				g_vn.push_back(vn->x);
				g_vn.push_back(vn->y);
				g_vn.push_back(vn->z);
			}

			if (mesh->HasTextureCoords(0)) {
				const aiVector3D* vt = &(mesh->mTextureCoords[0][v_i]);
				//printf ("      vt %i (%f,%f)\n", v_i, vt->x, vt->y);
				g_vt.push_back(vt->x);
				g_vt.push_back(vt->y);
			}



			if (mesh->HasTangentsAndBitangents()) {
				// NB: could store/print tangents here
			}
		}
	}

	aiReleaseImport(scene);
	return true;
}

void Object::generateObjectBufferMesh(GLuint shaderProgramID, int numVertices) {
	/*----------------------------------------------------------------------------
	LOAD MESH HERE AND COPY INTO BUFFERS
	----------------------------------------------------------------------------*/
	if (numVertices < 1) {
		std::cerr << "Error: The mesh dont have positions or normals" << std::endl;
		return;
	}
	
	//Note: you may get an error "vector subscript out of range" if you are using this code for a mesh that doesnt have positions and normals
	//Might be an idea to do a check for that before generating and binding the buffer.
	glGenBuffers(1, &this->vp_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vp_vbo);
	glBufferData(GL_ARRAY_BUFFER, numVertices * 3 * sizeof(float), &g_vp[0], GL_STATIC_DRAW);

	glGenBuffers(1, &this->vn_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vn_vbo);
	glBufferData(GL_ARRAY_BUFFER, numVertices * 3 * sizeof(float), &g_vn[0], GL_STATIC_DRAW);

	if (this->texture != NULL) {
		glGenBuffers(1, &this->vt_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vt_vbo);
		glBufferData(GL_ARRAY_BUFFER, numVertices * 2 * sizeof(float), &g_vt[0], GL_STATIC_DRAW);
	}
}

void Object::setVAO(float v) {
	this->vao = v;
}

void Object::setShader(Shader* s) {
	this->shader = s;
}

void Object::setVertexCounter(int n) {
	this->vertexCounter = n;
}

std::vector<float> Object::getVertices() {
	return this->g_vp;
}

void Object::setCenterBoudingBox(vec3 v) {
	this->centerBoundingBox = v;
}

void Object::setSizeBoudingBox(vec3 v) {
	this->sizeBoundingBox = v;
}

void Object::setVertices(std::vector<float> v) {
	this->g_vp = v;
}

void Object::setBoundingBox(GLuint vbo, GLuint elements, vec3 size, vec3 center, vec3 min, vec3 max) {
	this->vbo_vertices_bounding_box = vbo;
	this->ibo_elements = elements;
	this->sizeBoundingBox = size;
	this->centerBoundingBox = center;
	this->minBoundingBox = min;
	this->maxBoundingBox = max;
}

void Object::setVPVBO(GLuint v) {
	this->vp_vbo = v;
}

Object* Object::clone() {
	Object* obj = new Object();
	obj->setPosition(this->position);
	for (std::vector<Object*>::iterator it = this->childs.begin(); it != this->childs.end(); ++it) {
		obj->addChild(*it);
	}
	this->bindVBO();

	obj->setRotation(this->rotation);
	obj->setVAO(this->vao);
	obj->setShader(this->shader);
	obj->setVertexCounter(this->vertexCounter);

	obj->setVertices(this->g_vp);
	obj->setVPVBO(this->vp_vbo);
	obj->setBoundingBox(this->vbo_vertices_bounding_box, 
						this->ibo_elements,
						this->sizeBoundingBox,
						this->centerBoundingBox,
						this->minBoundingBox,
						this->maxBoundingBox);

	return obj;
}


void Object::generateBoundingBox() {
	if (this->getVertices().size() == 0)
		return;

	// Cube 1x1x1, centered on origin
	GLfloat vertices[] = {
		-0.5, -0.5, -0.5,
		0.5, -0.5, -0.5,
		0.5,  0.5, -0.5,
		-0.5,  0.5, -0.5,
		-0.5, -0.5,  0.5,
		0.5, -0.5,  0.5,
		0.5,  0.5,  0.5,
		-0.5,  0.5,  0.5,
	};
	glGenBuffers(1, &vbo_vertices_bounding_box);
	glEnableVertexAttribArray(loc1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices_bounding_box);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLushort elements[] = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		0, 4, 1, 5, 2, 6, 3, 7
	};

	glGenBuffers(1, &ibo_elements);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	this->updateBoundingBox();
}

void Object::updateBoundingBox(){
	GLfloat
		min_x, max_x,
		min_y, max_y,
		min_z, max_z;
	min_x = max_x = this->getVertices()[0];
	min_y = max_y = this->getVertices()[1];
	min_z = max_z = this->getVertices()[2];

	mat4 transform = identity_mat4();
	transform = rotate_z_deg(transform, rotation.v[2]);
	transform = rotate_y_deg(transform, rotation.v[1]);
	transform = rotate_x_deg(transform, rotation.v[0]);
	
	for (int i = 0; i < this->getVertices().size(); i += 3) {
		vec3 vertice = vec3(this->getVertices()[i],
						this->getVertices()[i + 1],
						this->getVertices()[i + 2]);
		vertice = vec3(transform * vec4(vertice, 1.0));
	
		if (vertice.v[0] < min_x) min_x = vertice.v[0];
		if (vertice.v[0] > max_x) max_x = vertice.v[0];
		if (vertice.v[1] < min_y) min_y = vertice.v[1];
		if (vertice.v[1] > max_y) max_y = vertice.v[1];
		if (vertice.v[2] < min_z) min_z = vertice.v[2];
		if (vertice.v[2] > max_z) max_z = vertice.v[2];
	}

	maxBoundingBox = vec3(max_x, max_y, max_z);
	minBoundingBox = vec3(min_x, min_y, min_z);
	sizeBoundingBox = vec3(max_x - min_x, max_y - min_y, max_z - min_z);
	centerBoundingBox = vec3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2);
}

void Object::display(mat4 view, mat4 projection,mat4 fatherMatrix){
	this->bindVBO();
	if (this->texture != NULL)
		this->texture->Bind(0);

	GLuint shaderProgramID = this->shader->getProgramID();
	glUseProgram(shaderProgramID);

	int matrix_location = glGetUniformLocation(shaderProgramID, "model");
	int view_mat_location = glGetUniformLocation(shaderProgramID, "view");
	int proj_mat_location = glGetUniformLocation(shaderProgramID, "proj");

	this->model = identity_mat4();

	this->model = scale(model, this->scaleVec);

	this->model = rotate_z_deg(this->model, rotation.v[2]);
	this->model = rotate_y_deg(this->model, rotation.v[1]);
	this->model = rotate_x_deg(this->model, rotation.v[0]);

	this->model = translate(this->model, position);

	mat4 resultMatrix = fatherMatrix*this->model;

	// update uniforms & draw
	glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, projection.m);
	glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, view.m);
	glUniformMatrix4fv(matrix_location, 1, GL_FALSE, resultMatrix.m);

	glDrawArrays(GL_TRIANGLES, 0, this->vertexCounter);

	int i = 0;
	for (std::vector<Object*>::iterator it = this->childs.begin(); it != this->childs.end(); ++it) {
		(*it)->display(view, projection, resultMatrix);
	}
}

void Object::display(mat4 mvp) {
	this->bindVBO();
	if (this->texture != NULL)
		this->texture->Bind(0);

	GLuint shaderProgramID = this->shader->getProgramID();
	glUseProgram(shaderProgramID);

	int matrix_location = glGetUniformLocation(shaderProgramID, "model");
	int view_mat_location = glGetUniformLocation(shaderProgramID, "view");
	int proj_mat_location = glGetUniformLocation(shaderProgramID, "proj");

	// update uniforms & draw
	glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, mvp.m);
	glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, identity_mat4().m);
	glUniformMatrix4fv(matrix_location, 1, GL_FALSE, identity_mat4().m);

	glDrawArrays(GL_TRIANGLES, 0, this->vertexCounter);
}

void Object::display(cv::Mat mvp) {
	this->bindVBO();
	if (this->texture != NULL)
		this->texture->Bind(0);

	GLuint shaderProgramID = this->shader->getProgramID();
	glUseProgram(shaderProgramID);

	int matrix_location = glGetUniformLocation(shaderProgramID, "model");
	int view_mat_location = glGetUniformLocation(shaderProgramID, "view");
	int proj_mat_location = glGetUniformLocation(shaderProgramID, "proj");

	mat4 projection = mat4();
	mat4 view = mat4();
	// update uniforms & draw
	glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, projection.m);
	glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, view.m);
	glUniformMatrix4fv(matrix_location, 1, GL_FALSE, view.m);

	glDrawArrays(GL_TRIANGLES, 0, this->vertexCounter);
}


void Object::drawBoundingBox() {
	/* Apply object's transformation matrix */
	//updateBoundingBox();
	mat4 transform = translate(identity_mat4(), centerBoundingBox)* scale(identity_mat4(), sizeBoundingBox);
	transform = scale(transform, this->scaleVec);

	//transform = rotate_z_deg(transform, rotation.v[2]);
	//transform = rotate_y_deg(transform, rotation.v[1]);
	//transform = rotate_x_deg(transform, rotation.v[0]);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices_bounding_box);

	loc1 = glGetAttribLocation(this->shader->getProgramID(), "vertex_position");
	glVertexAttribPointer(
		loc1,  // attribute
		3,                  // number of elements per vertex, here (x,y,z)
		GL_FLOAT,           // the type of each element
		GL_FALSE,           // take our values as-is
		0,                  // no extra data between each position
		NULL
		);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
	
	glUseProgram(this->shader->getProgramID());
	
	transform = translate(transform, this->getPosition());
	int matrix_location = glGetUniformLocation(this->shader->getProgramID(), "model");
	glUniformMatrix4fv(matrix_location, 1, GL_FALSE, transform.m);

	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, 0);
	glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (GLvoid*)(4 * sizeof(GLushort)));
	glDrawElements(GL_LINES, 8, GL_UNSIGNED_SHORT, (GLvoid*)(8 * sizeof(GLushort)));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//glDisableVertexAttribArray(loc1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glDeleteBuffers(1, &vbo_vertices_bounding_box);
	//glDeleteBuffers(1, &ibo_elements);
}

vec3 Object::getMinVectorInWorld() {
	mat4 transform = identity_mat4();
	//mat4 transform = translate(identity_mat4(), centerBoundingBox)* scale(identity_mat4(), sizeBoundingBox);
	//mat4 transform = translate(identity_mat4(), centerBoundingBox);
	transform = scale(transform, this->scaleVec);

	transform = translate(transform, this->getPosition());

	return vec3(transform* vec4(this->minBoundingBox, 1.0));
}

vec3 Object::getMaxVectorInWorld() {
	mat4 transform = identity_mat4();
	//mat4 transform = translate(identity_mat4(), centerBoundingBox)* scale(identity_mat4(), sizeBoundingBox);
	//mat4 transform = translate(identity_mat4(), centerBoundingBox);

	transform = translate(transform, this->getPosition());

	return vec3(transform* vec4(this->maxBoundingBox, 1.0));
}

bool Object::checkCollision(Object* object){
	vec3 maxInWorld = this->getMaxVectorInWorld(); 
    vec3 minInWorld = this->getMinVectorInWorld(); 
	assert( length(maxInWorld) > 0);
	assert( length(minInWorld) > 0);
	assert( length(object->getMaxVectorInWorld()) > 0);
	assert( length(object->getMinVectorInWorld()) > 0);

	return(maxInWorld.v[0] >= object->getMinVectorInWorld().v[0] &&
    minInWorld.v[0] < object->getMaxVectorInWorld().v[0] &&
    maxInWorld.v[1] >= object->getMinVectorInWorld().v[1] &&
    minInWorld.v[1] < object->getMaxVectorInWorld().v[1] &&
    maxInWorld.v[2] >= object->getMinVectorInWorld().v[2] &&
    minInWorld.v[2] < object->getMaxVectorInWorld().v[2]);
}
