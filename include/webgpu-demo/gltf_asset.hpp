#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <string>
#include <vector>
#include <optional>

namespace glTF
{

    struct Buffer
    {
        std::string uri;
        std::uint32_t byteLength;
    };

    struct BufferView
    {
        std::uint32_t buffer;
        std::uint32_t byteOffset;
        std::uint32_t byteLength;
        std::optional<std::uint32_t> byteStride;
    };

    struct Accessor
    {
        enum class ComponentType : std::uint32_t
        {
            Byte = 5120,
            UnsignedByte = 5121,
            Short = 5122,
            UnsignedShort = 5123,
            UnsignedInt = 5125,
            Float = 5126,
        };

        enum class Type : std::uint32_t
        {
            Scalar,
            Vec2,
            Vec3,
            Vec4,
            Mat2,
            Mat3,
            Mat4,
            Unknown,
        };

        std::uint32_t bufferView;
        std::uint32_t byteOffset;
        ComponentType componentType;
        bool normalized;
        std::uint32_t count;
        Type type;
    };

    struct Primitive
    {
        struct Attributes
        {
            std::optional<std::uint32_t> position;
            std::optional<std::uint32_t> normal;
            std::optional<std::uint32_t> tangent;
            std::optional<std::uint32_t> texcoord;
        };

        enum class Mode : std::uint32_t
        {
            Points = 0,
            Lines = 1,
            LineLoop = 2,
            LineStrip = 3,
            Triangles = 4,
            TriangleStrip = 5,
            TriangleFan = 6,
        };

        Attributes attributes;
        std::optional<std::uint32_t> indices;
        Mode mode;
    };

    struct Mesh
    {
        std::vector<Primitive> primitives;
    };

    struct Node
    {
        std::string name;

        glm::vec3 translation;
        glm::quat rotation;
        glm::vec3 scale;

        std::vector<std::uint32_t> children;
        std::optional<std::uint32_t> mesh;
    };

    struct Asset
    {
        std::vector<Node> nodes;
        std::vector<Mesh> meshes;
        std::vector<Accessor> accessors;
        std::vector<BufferView> bufferViews;
        std::vector<Buffer> buffers;
    };

}
