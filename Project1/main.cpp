#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aTangent;
layout(location = 4) in vec3 aBitangent;

uniform mat4 mvp;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out mat3 TBN;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal  = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    vec3 T = normalize(mat3(model) * aTangent);
    vec3 B = normalize(mat3(model) * aBitangent);
    vec3 N = normalize(Normal);
    TBN = mat3(T, B, N);

    gl_Position = mvp * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in mat3 TBN;

out vec4 FragColor;

uniform vec4 color;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D normalMap;

void main() {
    vec3 norm = texture(normalMap, TexCoord).rgb;
    norm = norm * 2.0 - 1.0;
    norm = normalize(TBN * norm);

    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * vec3(1.0);

    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);

    float specularStrength = 1.0;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 128.0);
    vec3 specular = specularStrength * spec * vec3(1.0);

    vec3 result = (ambient + diffuse + specular) * color.rgb;
    FragColor = vec4(result, color.a);
}
)";

glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
float yaw = -135.0f;
float pitch = -35.0f;
float lastX = 500.0f;
float lastY = 500.0f;
bool  firstMouse = true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
    }

    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    lastX = (float)xpos;
    lastY = (float)ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)  pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return shader;
}


void computeTangents(
    const std::vector<float>& vertices,
    const std::vector<unsigned int>& indices,
    std::vector<glm::vec3>& out_tangents,
    std::vector<glm::vec3>& out_bitangents)
{
    int vertexCount = (int)(vertices.size() / 8);
    out_tangents.assign(vertexCount, glm::vec3(0.0f));
    out_bitangents.assign(vertexCount, glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        unsigned int i0 = indices[i];
        unsigned int i1 = indices[i + 1];
        unsigned int i2 = indices[i + 2];

        auto getPos = [&](unsigned int idx) {
            return glm::vec3(
                vertices[idx * 8 + 0],
                vertices[idx * 8 + 1],
                vertices[idx * 8 + 2]);
            };
        auto getUV = [&](unsigned int idx) {
            return glm::vec2(
                vertices[idx * 8 + 6],
                vertices[idx * 8 + 7]);
            };

        glm::vec3 p0 = getPos(i0);
        glm::vec3 p1 = getPos(i1);
        glm::vec3 p2 = getPos(i2);

        glm::vec2 uv0 = getUV(i0);
        glm::vec2 uv1 = getUV(i1);
        glm::vec2 uv2 = getUV(i2);

        glm::vec3 deltaPos1 = p1 - p0;
        glm::vec3 deltaPos2 = p2 - p0;

        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        float r = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if (fabs(r) < 1e-6f) continue;
        r = 1.0f / r;

        glm::vec3 T = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
        glm::vec3 B = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;

        out_tangents[i0] += T;
        out_tangents[i1] += T;
        out_tangents[i2] += T;

        out_bitangents[i0] += B;
        out_bitangents[i1] += B;
        out_bitangents[i2] += B;
    }

    for (int i = 0; i < vertexCount; ++i) {
        if (glm::length(out_tangents[i]) > 1e-6f)
            out_tangents[i] = glm::normalize(out_tangents[i]);
        else
            out_tangents[i] = glm::vec3(1.0f, 0.0f, 0.0f);


        if (glm::length(out_bitangents[i]) > 1e-6f)
            out_bitangents[i] = glm::normalize(out_bitangents[i]);
        else
            out_bitangents[i] = glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

GLuint loadTexture(const char* path) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_set_flip_vertically_on_load(true);

    int w, h, nrCh;
    unsigned char* data = stbi_load(path, &w, &h, &nrCh, 0);
    if (data) {
        GLenum format = GL_RGB;
        if (nrCh == 4) format = GL_RGBA;
        else if (nrCh == 1) format = GL_RED;

        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        std::cout << "Texture loaded: " << path << " (" << w << "x" << h << ", " << nrCh << " ch)\n";
    }
    else {
        std::cerr << "Failed to load texture: " << path << "\n";
    }
    stbi_image_free(data);
    return tex;
}

int main() {
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int width = 1000;
    int height = 1000;
    GLFWwindow* window = glfwCreateWindow(width, height, "Bump Mapping", nullptr, nullptr);
    if (!window) { std::cerr << "Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    float S = 0.3f;

    std::vector<float> cubeVerts;
    std::vector<unsigned int> cubeIdx;

    auto addFace = [&](glm::vec3 normal,
        glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
        {
            unsigned int start = (unsigned int)(cubeVerts.size() / 8);

            glm::vec2 uvs[4] = {
                {0.0f, 0.0f},
                {1.0f, 0.0f},
                {1.0f, 1.0f},
                {0.0f, 1.0f}
            };
            glm::vec3 pos[4] = { p0, p1, p2, p3 };

            for (int i = 0; i < 4; ++i) {
                cubeVerts.push_back(pos[i].x);
                cubeVerts.push_back(pos[i].y);
                cubeVerts.push_back(pos[i].z);
                cubeVerts.push_back(normal.x);
                cubeVerts.push_back(normal.y);
                cubeVerts.push_back(normal.z);
                cubeVerts.push_back(uvs[i].x);
                cubeVerts.push_back(uvs[i].y);
            }

            cubeIdx.push_back(start + 0);
            cubeIdx.push_back(start + 1);
            cubeIdx.push_back(start + 2);
            cubeIdx.push_back(start + 0);
            cubeIdx.push_back(start + 2);
            cubeIdx.push_back(start + 3);
        };

    addFace(glm::vec3(0, 0, 1),
        glm::vec3(-S, -S, S), glm::vec3(S, -S, S),
        glm::vec3(S, S, S), glm::vec3(-S, S, S));
    addFace(glm::vec3(0, 0, -1),
        glm::vec3(S, -S, -S), glm::vec3(-S, -S, -S),
        glm::vec3(-S, S, -S), glm::vec3(S, S, -S));
    addFace(glm::vec3(-1, 0, 0),
        glm::vec3(-S, -S, -S), glm::vec3(-S, -S, S),
        glm::vec3(-S, S, S), glm::vec3(-S, S, -S));
    addFace(glm::vec3(1, 0, 0),
        glm::vec3(S, -S, S), glm::vec3(S, -S, -S),
        glm::vec3(S, S, -S), glm::vec3(S, S, S));
    addFace(glm::vec3(0, -1, 0),
        glm::vec3(-S, -S, -S), glm::vec3(S, -S, -S),
        glm::vec3(S, -S, S), glm::vec3(-S, -S, S));
    addFace(glm::vec3(0, 1, 0),
        glm::vec3(-S, S, S), glm::vec3(S, S, S),
        glm::vec3(S, S, -S), glm::vec3(-S, S, -S));

    std::vector<glm::vec3> cubeTangents, cubeBitangents;
    computeTangents(cubeVerts, cubeIdx, cubeTangents, cubeBitangents);

    std::vector<float> cubeData;
    int cubeVertexCount = (int)(cubeVerts.size() / 8);
    for (int i = 0; i < cubeVertexCount; ++i) {
        for (int k = 0; k < 8; ++k)
            cubeData.push_back(cubeVerts[i * 8 + k]);
        cubeData.push_back(cubeTangents[i].x);
        cubeData.push_back(cubeTangents[i].y);
        cubeData.push_back(cubeTangents[i].z);
        cubeData.push_back(cubeBitangents[i].x);
        cubeData.push_back(cubeBitangents[i].y);
        cubeData.push_back(cubeBitangents[i].z);
    }

    float Py = 0.3f;
    float Pa = 0.3f;
    float Ph = 0.8f;

    glm::vec3 apex(0.0f, Ph, 0.0f);
    glm::vec3 base[4] = {
        glm::vec3(-Pa, Py, -Pa),
        glm::vec3(Pa, Py, -Pa),
        glm::vec3(Pa, Py,  Pa),
        glm::vec3(-Pa, Py,  Pa),
    };

    std::vector<float> pyrVerts; 
    std::vector<unsigned int> pyrIdx;

    {
        glm::vec3 n(0, -1, 0);
        glm::vec3 positions[4] = { base[0], base[1], base[2], base[3] };
        unsigned int order[6] = { 0,2,1, 0,3,2 };
        glm::vec2 uvs[4] = {
            {0,0}, {1,0}, {1,1}, {0,1}
        };
        unsigned int start = (unsigned int)(pyrVerts.size() / 8);
        for (int i = 0; i < 4; ++i) {
            pyrVerts.push_back(positions[i].x);
            pyrVerts.push_back(positions[i].y);
            pyrVerts.push_back(positions[i].z);
            pyrVerts.push_back(n.x);
            pyrVerts.push_back(n.y);
            pyrVerts.push_back(n.z);
            pyrVerts.push_back(uvs[i].x);
            pyrVerts.push_back(uvs[i].y);
        }
        for (int i = 0; i < 6; ++i)
            pyrIdx.push_back(start + order[i]);
    }

    unsigned int sides[4][2] = { {0,1}, {1,2}, {2,3}, {3,0} };
    glm::vec2 sideUVs[3] = { {0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 1.0f} };
    for (int i = 0; i < 4; ++i) {
        glm::vec3 a = base[sides[i][0]];
        glm::vec3 b = base[sides[i][1]];
        glm::vec3 n = -glm::normalize(glm::cross(b - a, apex - a));

        unsigned int start = (unsigned int)(pyrVerts.size() / 8);
        glm::vec3 tri[3] = { a, b, apex };
        for (int j = 0; j < 3; ++j) {
            pyrVerts.push_back(tri[j].x);
            pyrVerts.push_back(tri[j].y);
            pyrVerts.push_back(tri[j].z);
            pyrVerts.push_back(n.x);
            pyrVerts.push_back(n.y);
            pyrVerts.push_back(n.z);
            pyrVerts.push_back(sideUVs[j].x);
            pyrVerts.push_back(sideUVs[j].y);
        }
        pyrIdx.push_back(start + 0);
        pyrIdx.push_back(start + 1);
        pyrIdx.push_back(start + 2);
    }

    std::vector<glm::vec3> pyrTangents, pyrBitangents;
    computeTangents(pyrVerts, pyrIdx, pyrTangents, pyrBitangents);

    std::vector<float> pyrData;
    int pyrVertexCount = (int)(pyrVerts.size() / 8);
    for (int i = 0; i < pyrVertexCount; ++i) {
        for (int k = 0; k < 8; ++k)
            pyrData.push_back(pyrVerts[i * 8 + k]);
        pyrData.push_back(pyrTangents[i].x);
        pyrData.push_back(pyrTangents[i].y);
        pyrData.push_back(pyrTangents[i].z);
        pyrData.push_back(pyrBitangents[i].x);
        pyrData.push_back(pyrBitangents[i].y);
        pyrData.push_back(pyrBitangents[i].z);
    }

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, cubeData.size() * sizeof(float), cubeData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cubeIdx.size() * sizeof(unsigned int), cubeIdx.data(), GL_STATIC_DRAW);

    int stride = 14 * sizeof(float);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);

    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, (void*)(11 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glBindVertexArray(0);

    GLuint VAO_pyr, VBO_pyr, EBO_pyr;
    glGenVertexArrays(1, &VAO_pyr);
    glGenBuffers(1, &VBO_pyr);
    glGenBuffers(1, &EBO_pyr);

    glBindVertexArray(VAO_pyr);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_pyr);
    glBufferData(GL_ARRAY_BUFFER, pyrData.size() * sizeof(float), pyrData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pyr);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, pyrIdx.size() * sizeof(unsigned int), pyrIdx.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, (void*)(11 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glBindVertexArray(0);

    int pyrIndexCount = (int)pyrIdx.size();

    GLuint normalMap = loadTexture("texture1.jpg");

    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        (float)width / (float)height,
        0.1f, 100.0f
    );

    glm::mat4 model_cube = glm::mat4(1.0f);
    glm::mat4 model_pyr = glm::mat4(1.0f);

    glm::vec3 cameraPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
    }

    GLuint mvpLoc = glGetUniformLocation(program, "mvp");
    GLuint modelLoc = glGetUniformLocation(program, "model");
    GLuint colorLoc = glGetUniformLocation(program, "color");
    GLuint lightPosLoc = glGetUniformLocation(program, "lightPos");
    GLuint viewPosLoc = glGetUniformLocation(program, "viewPos");
    GLuint normalMapLoc = glGetUniformLocation(program, "normalMap");

    glm::vec3 lightPos = glm::vec3(2.0f, 3.0f, 2.0f);

    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        float currentFrame = (float)glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        float speed = 2.0f * deltaTime;
        glm::vec3 flatFront(cameraFront.x, 0.0f, cameraFront.z);
        if (glm::length(flatFront) > 0.001f)
            flatFront = glm::normalize(flatFront);
        glm::vec3 right = glm::normalize(glm::cross(flatFront, cameraUp));

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraPos += speed * flatFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= speed * flatFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= speed * right;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraPos += speed * right;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) cameraPos.y += speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) cameraPos.y -= speed;

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);

        glUniform3f(lightPosLoc, lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(viewPosLoc, cameraPos.x, cameraPos.y, cameraPos.z);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, normalMap);
        glUniform1i(normalMapLoc, 0);

        // Куб
        glUniform4f(colorLoc, 0.2f, 0.6f, 1.0f, 1.0f);
        glm::mat4 mvp_cube = proj * view * model_cube;
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp_cube));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model_cube));
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)cubeIdx.size(), GL_UNSIGNED_INT, 0);

        // Пирамида
        glUniform4f(colorLoc, 1.0f, 0.5f, 0.1f, 1.0f);
        glm::mat4 mvp_pyr = proj * view * model_pyr;
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp_pyr));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model_pyr));
        glBindVertexArray(VAO_pyr);
        glDrawElements(GL_TRIANGLES, pyrIndexCount, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteTextures(1, &normalMap);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO_pyr);
    glDeleteBuffers(1, &VBO_pyr);
    glDeleteBuffers(1, &EBO_pyr);
    glDeleteProgram(program);
    glfwTerminate();

    return 0;
}