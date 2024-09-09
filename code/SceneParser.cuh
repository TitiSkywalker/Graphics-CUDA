/* SceneParser stays inside CPU, but holds a bunch of pointers pointing toward GPU.
* The format of scene file is basically a context-free language:
* 
*   [section name] { (optional section size) content }
* 
* where the section name can be:
* 
*   - PerspectiveCamera
*       - center    <float> x3
*       - direction <float> x3
*       - up        <float> x3
*       - angle     <float>
* 
*   - Background
*       - color     <float> x3
* 
*   - Lights:
*       - numLights                 <int>
*       - [light type] { content }
*       for light type = DirectionalLight:
*           - direction             <float> x3
*           - color                 <float x3
*       for light type = PointLight:
*           - center                <float> x3
*           - color                 <float> x3
*           - (optional falloff)    <float>
* 
*   - Materials:
*       - numMaterials              <int>
*       - [material type] { content }
*       material type can be Material/PhongMaterial/Ambient/Glossy/Mirror/Emit
* 
*   - Group:
*       - numObjects                <int>
*       - MaterialIndex             <int> start from 0, maintain last material if not specified
*       - [object type] { content }
*       object type can be Plane/Sphere/Triangle(counter clockwise for computing normal)
*/
#pragma once
#include <cassert>
#include <vector>
#include <stdexcept>
#include <string>

#include "Vecmath.cuh"
#include "Tool.cuh"
#include "File.cuh"

#include "Camera.cuh"
#include "Material.cuh"
#include "Object3D.cuh"
#include "Sphere.cuh"
#include "Plane.cuh"
#include "Triangle.cuh"
#include "Velocity.cuh"
#include "Light.cuh"

#define MAX_PARSER_TOKEN_LENGTH 100

using namespace std;

class SceneParser
{
public:
    SceneParser(const char* filename)
    {
        cout << "Read in scene file " << filename << endl;
        //initialize some reasonable default values
        group = NULL;
        camera = NULL;
        background = NULL;
        lights = NULL;
        lightObjects = NULL;
        ambient_light = Vector3f(0, 0, 0);
        num_objects = 0;
        num_materials = 0;
        num_lights = 0;
        num_lightobjects = 0;
        materials = NULL;
        materialTypes = NULL;
        currentMaterialType = MATERIAL;
        currentMaterial = NULL;

        try
        {
            //parse the file
            if (filename == NULL)
                throw runtime_error("filename is NULL");

            const char* ext = &filename[strlen(filename) - 6];

            if (strcmp(ext, ".scene") != 0)
                throw runtime_error("wrong file name extension: " + string(ext));

            string filePath = getInputFilePath(filename);
            file = fopen(filePath.c_str(), "r");

            if (file == NULL)
                throw runtime_error("cannot open scene file: " + string(filename));

            parseFile();
            fclose(file);
            file = NULL;

            if (background == NULL)
                background = Tool::newBackground(Vector3f(0, 0, 0));

            if (num_lightobjects > 0)
                prepareLightGroup();

            Tool::deviceSynchronize();

            printf("Total %d objects, %d materials, %d lights, %d light objects\n",
                num_objects, num_materials, num_lights, num_lightobjects);
        }
        catch (const runtime_error& e)
        {
            everythingOK = false;
            errorMessage = e.what();
        }
    }

    ~SceneParser()
    {
        if (background != NULL)
            Tool::cudaFreeChecked(background, "SceneParser::~SceneParser() -freeing background");
        if (camera != NULL)
            Tool::cudaFreeChecked(camera, "SceneParser::~SceneParser() -freeing camera");
        if (group != NULL)
        {
            for (auto object : objectPool)
                Tool::cudaFreeChecked(object, "Sceneparser::~SceneParser() -freeing group");
            delete[] group;
            delete[] isLightObject;
        }
        if (lightObjects != NULL)
        {
            //pointers in this array is a subset of "group"
            delete[] lightObjects;
        }
        if (materials != NULL)
        {
            for (int i = 0; i < num_materials; i++)
            {
                if (materials[i] != NULL)
                    Tool::cudaFreeChecked(materials[i], "SceneParser::~SceneParser() -freeing materials");
            }
            delete[] materials;
            delete[] materialTypes;
        }
        if (lights != NULL)
        {
            for (int i = 0; i < num_lights; i++)
            {
                if (lights[i] != NULL)
                    Tool::cudaFreeChecked(lights[i], "SceneParser::~SceneParser() -freeing lights");
            }
            delete[] lights;
        }
    }

    Camera* getCamera() const
    {
        return camera;
    }

    Background* getBackground() const
    {
        return background;
    }

    Vector3f getAmbientLight() const
    {
        return ambient_light;
    }

    int getNumMaterials() const
    {
        return num_materials;
    }

    Material** getMaterials() const
    {
        return materials;
    }

    int getNumObjects() const
    {
        return num_objects;
    }

    Object3D** getGroup() const
    {
        return group;
    }

    int getNumLights() const
    {
        return num_lights;
    }

    Light** getLights() const
    {
        return lights;
    }

    int getNumLightObjects() const
    {
        return num_lightobjects;
    }

    Object3D** getLightObjects() const
    {
        return lightObjects;
    }

    bool checkStatus() const
    {
        return everythingOK;
    }

    string getErrorMessage() const
    {
        return errorMessage;
    }
private:

    void parseFile()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        while (getToken(token))
        {
            if (!strcmp(token, "PerspectiveCamera"))
            {
                parsePerspectiveCamera();
            }
            else if (!strcmp(token, "Background"))
            {
                parseBackground();
            }
            else if (!strcmp(token, "Lights"))
            {
                parseLights();
            }
            else if (!strcmp(token, "Materials"))
            {
                parseMaterials();
            }
            else if (!strcmp(token, "Group"))
            {
                parseGroup();
            }
            else
            {
                throw runtime_error("unknown token in parseFile: "+string(token));
            }
        }
    }

    void parsePerspectiveCamera()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        // read in the camera parameters
        getToken(token); matchToken(token, "{");

        getToken(token); matchToken(token, "center");
        Vector3f center = readVector3f();

        getToken(token); matchToken(token, "direction");
        Vector3f direction = readVector3f();

        getToken(token); matchToken(token, "up");
        Vector3f vertical = readVector3f();

        getToken(token); matchToken(token, "angle");
        float angle_degrees = readFloat();
        float angle_radians = ((float)M_PI * angle_degrees) / 180.0f;

        getToken(token); matchToken(token, "}");

        camera = Tool::newPerspectiveCamera(center, direction, vertical, angle_radians);
    }

    void parseBackground()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        // read in the background color
        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "}"))
            {
                break;
            }
            else if (!strcmp(token, "color"))
            {
                background = Tool::newBackground(readVector3f());
            }
            else if (!strcmp(token, "ambientLight"))
            {
                ambient_light = readVector3f();
            }
            else
            {
                throw runtime_error("unknown token in parseBackground: " + string(token));
            }
        }
    }

    void parseMaterials()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");

        // read in the number of objects
        getToken(token); matchToken(token, "numMaterials");
        num_materials = readInt();
        if (num_materials <= 0)
            throw runtime_error("numMaterials <= 0");

        //buffer stores pointers temporarily
        materials = new Material * [num_materials];
        materialTypes = new material_type[num_materials];

        //read in the objects
        int count = 0;
        while (num_materials > count)
        {
            getToken(token);
            if (!strcmp(token, "PhongMaterial") || (!strcmp(token, "Material")))
            {
                materials[count] = parsePhongMaterial();
                materialTypes[count] = PHONG;
            }
            else if (!strcmp(token, "Ambient"))
            {
                materials[count] = parseAmbientMaterial();
                materialTypes[count] = AMBIENT;
            }
            else if (!strcmp(token, "Glossy"))
            {
                materials[count] = parseGlossyMaterial();
                materialTypes[count] = GLOSSY;
            }
            else if (!strcmp(token, "Mirror"))
            {
                materials[count] = parseMirror();
                materialTypes[count] = MIRROR;
            }
            else if (!strcmp(token, "Emit"))
            {
                materials[count] = parseEmit();
                materialTypes[count] = EMIT;
            }
            else
            {
                throw runtime_error("unknown token in parseMaterial: " + string(token));
            }
            count++;
        }
        getToken(token); matchToken(token, "}");
    }
    Material* parsePhongMaterial()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Vector3f diffuseColor(1, 1, 1), specularColor(0, 0, 0);

        float shininess = 0;
        float refractionIndex = 0;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "diffuseColor"))
            {
                diffuseColor = readVector3f();
            }
            else if (!strcmp(token, "specularColor"))
            {
                specularColor = readVector3f();
            }
            else if (!strcmp(token, "shininess"))
            {
                shininess = readFloat();
            }
            else if (!strcmp(token, "refractionIndex"))
            {
                refractionIndex = readFloat();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }

        return Tool::newPhong(diffuseColor, specularColor, shininess, refractionIndex);
    }
    Material* parseAmbientMaterial()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Vector3f diffuseColor(1, 1, 1), specularColor(0, 0, 0);
        float shininess = 0;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "diffuseColor"))
            {
                diffuseColor = readVector3f();
            }
            else if (!strcmp(token, "specularColor"))
            {
                specularColor = readVector3f();
            }
            else if (!strcmp(token, "shininess"))
            {
                shininess = readFloat();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }
        return Tool::newAmbient(diffuseColor, specularColor, shininess);
    }
    Material* parseGlossyMaterial()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Vector3f diffuseColor(1, 1, 1), specularColor(0, 0, 0);

        float shininess = 0;
        float roughness = 0.5;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "diffuseColor"))
            {
                diffuseColor = readVector3f();
            }
            else if (!strcmp(token, "specularColor"))
            {
                specularColor = readVector3f();
            }
            else if (!strcmp(token, "shininess"))
            {
                shininess = readFloat();
            }
            else if (!strcmp(token, "roughness"))
            {
                roughness = readFloat();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }

        return Tool::newGlossy(diffuseColor, specularColor, shininess, roughness);
    }
    Material* parseMirror()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];

        getToken(token); matchToken(token, "{");
        getToken(token); matchToken(token, "}");

        return Tool::newMirror();
    }
    Material* parseEmit()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Vector3f color(0, 0, 0);

        float falloff = 0;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "color"))
            {
                color = readVector3f();
            }
            else if (!strcmp(token, "falloff"))
            {
                falloff = readFloat();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }

        return Tool::newEmit(color, falloff);
    }

    void parseLights()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");

        // read in the number of objects
        getToken(token); matchToken(token, "numLights");
        num_lights = readInt();
        if (num_lights <= 0)
        {
            num_lights = 0;
            return;
        }
        lights = new Light * [num_lights];

        // read in the objects
        int count = 0;
        while (num_lights > count)
        {
            getToken(token);
            if (!strcmp(token, "DirectionalLight"))
            {
                lights[count] = parseDirectionalLight();
            }
            else if (!strcmp(token, "PointLight"))
            {
                lights[count] = parsePointLight();
            }
            else
            {
                throw runtime_error("unknown token in parseLight: " + string(token));
            }
            count++;
        }
        getToken(token); matchToken(token, "}");
    }
    DirectionalLight* parseDirectionalLight()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");

        getToken(token); matchToken(token, "direction");
        Vector3f direction = readVector3f();
        getToken(token); matchToken(token, "color");
        Vector3f color = readVector3f();

        getToken(token); matchToken(token, "}");
        return Tool::newDirectionalLight(direction, color);
    }
    PointLight* parsePointLight()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];

        Vector3f position, color;
        float falloff = 0;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "position"))
            {
                position = readVector3f();
            }
            else if (!strcmp(token, "color"))
            {
                color = readVector3f();
            }
            else if (!strcmp(token, "falloff"))
            {
                falloff = readFloat();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }
        return Tool::newPointLight(position, color, falloff);
    }

    void parseGroup()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");

        // read in the number of objects
        getToken(token); matchToken(token, "numObjects");

        num_objects = readInt();

        if (num_objects > 0)
        {
            group = new Object3D * [num_objects];
            isLightObject = new bool[num_objects];
            for (int i = 0; i < num_objects; i++)
            {
                group[i] = NULL;
                isLightObject[i] = false;
            }
        }

        // read in the objects
        int count = 0;
        while (num_objects > count)
        {
            getToken(token);
            if (!strcmp(token, "MaterialIndex"))
            {
                // change the current material
                int index = readInt();
                if (index < 0 && index >= num_materials)
                    throw runtime_error("material index out of range");
                currentMaterial = materials[index];
                currentMaterialType = materialTypes[index];
            }
            else
            {
                Object3D* object = parseObject(token);
                if (object == NULL)
                    throw runtime_error("a NULL object is produced in parseGroup");
                group[count] = object;
                if (currentMaterialType == EMIT) 
                {
                    num_lightobjects++;
                    isLightObject[count] = true;
                }
                count++;
            }
        }
        getToken(token); matchToken(token, "}");
    }
    Object3D* parseObject(char token[MAX_PARSER_TOKEN_LENGTH])
    {
        Object3D* answer = NULL;
        if (!strcmp(token, "Sphere"))
        {
            answer = (Object3D*)parseSphere();
        }
        else if (!strcmp(token, "Plane"))
        {
            answer = (Object3D*)parsePlane();
        }
        else if (!strcmp(token, "Triangle"))
        {
            answer = (Object3D*)parseTriangle();
        }
        else if (!strcmp(token, "Velocity"))
        {
            answer = (Object3D*)parseVelocity();
        }
        else
        {
            throw runtime_error("unknown token in parseObject: " + string(token));
        }

        objectPool.push_back(answer);

        return answer;
    }
    Sphere* parseSphere()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");
        getToken(token); matchToken(token, "center");
        Vector3f center = readVector3f();
        getToken(token); matchToken(token, "radius");
        float radius = readFloat();
        getToken(token); matchToken(token, "}");

        if (currentMaterial == NULL)
            throw runtime_error("material for sphere is not specified");
        return Tool::newSphere(center, radius, currentMaterial);
    }
    Plane* parsePlane()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        getToken(token); matchToken(token, "{");
        getToken(token); matchToken(token, "normal");
        Vector3f normal = readVector3f();
        getToken(token); matchToken(token, "offset");
        float offset = readFloat();
        getToken(token); matchToken(token, "}");

        if (currentMaterial == NULL)
            throw runtime_error("material for plane is not specified");
        return Tool::newPlane(normal, offset, currentMaterial);
    }
    Triangle* parseTriangle()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];

        Vector3f position, color;
        Vector3f a, b, c;

        getToken(token); matchToken(token, "{");
        while (1)
        {
            getToken(token);
            if (!strcmp(token, "vertex0"))
            {
                a = readVector3f();
            }
            else if (!strcmp(token, "vertex1"))
            {
                b = readVector3f();
            }
            else if (!strcmp(token, "vertex2"))
            {
                c = readVector3f();
            }
            else
            {
                matchToken(token, "}");
                break;
            }
        }
        return Tool::newTriangle(a, b, c, currentMaterial);
    }
    Velocity* parseVelocity()
    {
        char token[MAX_PARSER_TOKEN_LENGTH];
        Vector3f velocity;
        Object3D* object = NULL;

        getToken(token); matchToken(token, "{");
        getToken(token);

        while (1)
        {
            if (!strcmp(token, "velocity"))
            {
                velocity = readVector3f();
            }
            else
            {
                object = parseObject(token);
                break;
            }

            getToken(token);
        }


        if (object == NULL)
            throw runtime_error("a NULL object is produced in parseVelocity");
        getToken(token); matchToken(token, "}");

        return Tool::newVelocity(velocity, object, currentMaterial);
    }

    inline int getToken(char token[MAX_PARSER_TOKEN_LENGTH]) 
    {
        //for simplicity, tokens must be separated by whitespace
        //tokens starting with '#' will be ignored
        if (file == NULL)
            throw runtime_error("cannot get token, file is NULL");
        while (true)
        {
            int success = fscanf(file, "%s ", token);
            if (success == EOF)
            {
                token[0] = '\0';
                return 0;
            }
            else if (token[0] != '#')
                return 1;
        }
    }
    inline void matchToken(char token[MAX_PARSER_TOKEN_LENGTH], const char* target)
    {
        if (strcmp(token, target))
            throw runtime_error("unknown token " + string(token) + ", expect " + target);
    }
    inline Vector3f readVector3f()
    {
        float x, y, z;
        int count = fscanf(file, "%f %f %f", &x, &y, &z);
        if (count != 3)
        {
            printf("Error trying to read 3 floats to make a Vector3f\n");
            assert(0);
        }
        return Vector3f(x, y, z);
    }
    inline Vector2f readVector2f()
    {
        float u, v;
        int count = fscanf(file, "%f %f", &u, &v);
        if (count != 2)
        {
            printf("Error trying to read 2 floats to make a Vec2f\n");
            assert(0);
        }
        return Vector2f(u, v);
    }
    inline float readFloat() 
    {
        float answer;
        int count = fscanf(file, "%f", &answer);
        if (count != 1) {
            printf("Error trying to read 1 float\n");
            assert(0);
        }
        return answer;
    }
    inline int readInt() 
    {
        int answer;
        int count = fscanf(file, "%d", &answer);
        if (count != 1) 
        {
            printf("Error trying to read 1 int\n");
            assert(0);
        }
        return answer;
    }

    void prepareLightGroup()
    {
        //copy light objects into another array
        lightObjects = new Object3D * [num_lightobjects];
        int top = 0;
        for (int i = 0; i < num_objects; i++)
        {
            if (isLightObject[i])
                lightObjects[top++] = group[i];
        }
    }

    FILE* file;                             //point to CPU
    Camera* camera;                         //point to GPU
    Background* background;                 //point to GPU
    Vector3f ambient_light;

    int num_objects;
    int num_materials;
    int num_lights;
    int num_lightobjects;

    vector<Object3D*> objectPool;           //store all objects to frees
    Object3D** group;                       //array in CPU, point to GPU  
    Material** materials;                   //array in CPU, point to GPU
    Material* currentMaterial;              //point to GPU
    material_type* materialTypes;           //array in CPU
    material_type currentMaterialType;    
    Light** lights;                         //array in CPU, point to GPU
    bool* isLightObject;                    //array in CPU
    Object3D** lightObjects;                //array in CPU, point in GPU

    //error handling
    string errorMessage;
    bool everythingOK;
};