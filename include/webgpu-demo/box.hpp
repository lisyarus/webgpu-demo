#pragma once

#include <glm/glm.hpp>

struct Box
{
    glm::vec3 min{ std::numeric_limits<float>::infinity()};
    glm::vec3 max{-std::numeric_limits<float>::infinity()};

    Box & expand(glm::vec3 const & p)
    {
        min = glm::min(min, p);
        max = glm::max(max, p);
        return *this;
    }

    Box & expand(Box const & b)
    {
        min = glm::min(min, b.min);
        max = glm::max(max, b.max);
        return *this;
    }

    glm::vec3 diagonal() const
    {
        return max - min;
    }
};
