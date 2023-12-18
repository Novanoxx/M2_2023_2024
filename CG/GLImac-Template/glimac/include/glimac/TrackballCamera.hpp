#pragma once

#include "glm.hpp"

class TrackballCamera
{
    public:
        TrackballCamera()
        {
            m_fDistance = -10.f;
            m_fAngleX = 25.f;
            m_fAngleY = 0.f;
        }

        TrackballCamera(float dist, float angleX, float angleY)
        {
            m_fDistance = dist;
            m_fAngleX = angleX;
            m_fAngleY = angleY;
        }

        void moveFront(float delta)
        {
            m_fDistance += delta;
        }

        void rotateLeft(float degree)
        {
            m_fAngleY += degree;
        }

        void rotateUp(float degree)
        {
            m_fAngleX += degree;
        }

        glm::mat4 getViewMatrix() const
        {
            glm::mat4 mat = glm::mat4(1.f);
            mat = glm::translate(mat, glm::vec3(0, 0, m_fDistance));
            mat = glm::rotate(mat, glm::radians(m_fAngleX), glm::vec3(1, 0, 0));
            mat = glm::rotate(mat, glm::radians(m_fAngleY), glm::vec3(0, 1, 0));
            return mat;
        }

    private :
        float m_fDistance;
        float m_fAngleX;
        float m_fAngleY;
};