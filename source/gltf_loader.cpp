#include <webgpu-demo/gltf_loader.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <fstream>
#include <stdexcept>

namespace glTF
{

    Asset load(std::filesystem::path const & path)
    {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Failed to open " + path.string());

        rapidjson::IStreamWrapper inputWrapper(input);
        rapidjson::Document document;
        document.ParseStream(inputWrapper);

        if (document.HasParseError())
            throw std::runtime_error("Failed to parse " + path.string() + ": " + std::to_string((int)document.GetParseError()));

        Asset result;

        if (document.HasMember("nodes"))
        for (auto const & nodeIn : document["nodes"].GetArray())
        {
            auto & node = result.nodes.emplace_back();

            if (nodeIn.HasMember("name"))
                node.name = nodeIn["name"].GetString();

            if (nodeIn.HasMember("children"))
                for (auto const & childId : nodeIn["children"].GetArray())
                    node.children.push_back(childId.GetUint());

            if (nodeIn.HasMember("mesh"))
                node.mesh = nodeIn["mesh"].GetUint();

            node.translation = glm::vec3(0.f);
            node.rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
            node.scale = glm::vec3(1.f);

            if (nodeIn.HasMember("translation"))
            {
                auto const & translation = nodeIn["translation"].GetArray();
                for (int i = 0; i < 3; ++i)
                    node.translation[i] = translation[i].GetFloat();
            }

            if (nodeIn.HasMember("rotation"))
            {
                auto const & rotation = nodeIn["rotation"].GetArray();
                for (int i = 0; i < 4; ++i)
                    node.rotation[i] = rotation[i].GetFloat();
            }

            if (nodeIn.HasMember("scale"))
            {
                auto const & scale = nodeIn["scale"].GetArray();
                for (int i = 0; i < 3; ++i)
                    node.scale[i] = scale[i].GetFloat();
            }

            if (nodeIn.HasMember("extensions"))
            {
                auto const & extensions = nodeIn["extensions"];
                if (extensions.HasMember("KHR_lights_punctual"))
                {
                    node.light = extensions["KHR_lights_punctual"]["light"].GetUint();
                }
            }
        }

        if (document.HasMember("meshes"))
        for (auto const & meshIn : document["meshes"].GetArray())
        {
            auto & mesh = result.meshes.emplace_back();

            for (auto const & primitiveIn : meshIn["primitives"].GetArray())
            {
                auto & primitive = mesh.primitives.emplace_back();

                primitive.mode = Primitive::Mode::Triangles;
                if (primitiveIn.HasMember("mode"))
                    primitive.mode = static_cast<Primitive::Mode>(primitiveIn["mode"].GetUint());

                if (primitiveIn.HasMember("indices"))
                    primitive.indices = primitiveIn["indices"].GetUint();

                if (primitiveIn.HasMember("material"))
                    primitive.material = primitiveIn["material"].GetUint();

                auto const & attributes = primitiveIn["attributes"];

                if (attributes.HasMember("POSITION"))
                    primitive.attributes.position = attributes["POSITION"].GetUint();

                if (attributes.HasMember("NORMAL"))
                    primitive.attributes.normal = attributes["NORMAL"].GetUint();

                if (attributes.HasMember("TANGENT"))
                    primitive.attributes.tangent = attributes["TANGENT"].GetUint();

                if (attributes.HasMember("TEXCOORD_0"))
                    primitive.attributes.texcoord = attributes["TEXCOORD_0"].GetUint();
            }
        }

        if (document.HasMember("materials"))
        for (auto const & materialIn : document["materials"].GetArray())
        {
            auto & material = result.materials.emplace_back();

            material.baseColorFactor = glm::vec4(1.f);
            material.metallicFactor = 1.f;
            material.roughnessFactor = 1.f;
            material.emissiveFactor = glm::vec3(0.f);
            material.cloth = false;

            if (materialIn.HasMember("pbrMetallicRoughness"))
            {
                auto const & pbrMetallicRoughness = materialIn["pbrMetallicRoughness"];

                if (pbrMetallicRoughness.HasMember("baseColorTexture"))
                    material.baseColorTexture = pbrMetallicRoughness["baseColorTexture"]["index"].GetUint();

                if (pbrMetallicRoughness.HasMember("metallicRoughnessTexture"))
                    material.metallicRoughnessTexture = pbrMetallicRoughness["metallicRoughnessTexture"]["index"].GetUint();
            }

            if (materialIn.HasMember("normalTexture"))
                material.normalTexture = materialIn["normalTexture"]["index"].GetUint();

            if (materialIn.HasMember("occlusionTexture"))
                material.occlusionTexture = materialIn["occlusionTexture"]["index"].GetUint();

            if (materialIn.HasMember("extras"))
            {
                auto const & extras = materialIn["extras"];

                if (extras.HasMember("cloth") && extras["cloth"].IsBool())
                    material.cloth = extras["cloth"].GetBool();
            }
        }

        if (document.HasMember("images"))
        for (auto const & imageIn : document["images"].GetArray())
        {
            auto & image = result.images.emplace_back();
            image.uri = imageIn["uri"].GetString();
        }

        if (document.HasMember("textures"))
        for (auto const & textureIn : document["textures"].GetArray())
        {
            auto & texture = result.textures.emplace_back();
            if (textureIn.HasMember("source"))
                texture.source = textureIn["source"].GetUint();
        }

        if (document.HasMember("accessors"))
        for (auto const & accessorIn : document["accessors"].GetArray())
        {
            auto & accessor = result.accessors.emplace_back();

            accessor.bufferView = accessorIn["bufferView"].GetUint();

            accessor.byteOffset = 0;
            if (accessorIn.HasMember("byteOffset"))
                accessor.byteOffset = accessorIn["byteOffset"].GetUint();

            accessor.componentType = static_cast<Accessor::ComponentType>(accessorIn["componentType"].GetUint());

            accessor.normalized = false;
            if (accessorIn.HasMember("normalized"))
                accessor.normalized = accessorIn["normalized"].GetBool();

            accessor.count = accessorIn["count"].GetUint();

            std::string const typeStr = accessorIn["type"].GetString();
            if (typeStr == "SCALAR")
                accessor.type = Accessor::Type::Scalar;
            else if (typeStr == "VEC2")
                accessor.type = Accessor::Type::Vec2;
            else if (typeStr == "VEC3")
                accessor.type = Accessor::Type::Vec3;
            else if (typeStr == "VEC4")
                accessor.type = Accessor::Type::Vec4;
            else if (typeStr == "MAT2")
                accessor.type = Accessor::Type::Mat2;
            else if (typeStr == "MAT3")
                accessor.type = Accessor::Type::Mat3;
            else if (typeStr == "MAT4")
                accessor.type = Accessor::Type::Mat4;
            else
                accessor.type = Accessor::Type::Unknown;
        }

        if (document.HasMember("bufferViews"))
        for (auto const & bufferViewIn : document["bufferViews"].GetArray())
        {
            auto & bufferView = result.bufferViews.emplace_back();

            bufferView.buffer = bufferViewIn["buffer"].GetUint();

            bufferView.byteOffset = 0;
            if (bufferViewIn.HasMember("byteOffset"))
                bufferView.byteOffset = bufferViewIn["byteOffset"].GetUint();

            bufferView.byteLength = bufferViewIn["byteLength"].GetUint();

            if (bufferViewIn.HasMember("byteStride"))
                bufferView.byteStride = bufferViewIn["byteStride"].GetUint();
        }

        if (document.HasMember("buffers"))
        for (auto const & bufferIn : document["buffers"].GetArray())
        {
            auto & buffer = result.buffers.emplace_back();

            buffer.uri = bufferIn["uri"].GetString();
            buffer.byteLength = bufferIn["byteLength"].GetUint();
        }

        if (document.HasMember("extensions"))
        {
            auto const & extensions = document["extensions"];

            if (extensions.HasMember("KHR_lights_punctual"))
            {
                for (auto const & lightIn : extensions["KHR_lights_punctual"]["lights"].GetArray())
                {
                    auto & light = result.lights.emplace_back();

                    auto const & colorIn = lightIn["color"].GetArray();
                    light.intensity.r = colorIn[0].GetFloat();
                    light.intensity.g = colorIn[1].GetFloat();
                    light.intensity.b = colorIn[2].GetFloat();
                    light.intensity *= lightIn["intensity"].GetFloat();
                }
            }
        }

        return result;
    }

    std::vector<char> loadBuffer(std::filesystem::path const & assetPath, std::string const & bufferUri)
    {
        auto const bufferPath = assetPath.parent_path() / bufferUri;

        std::vector<char> result;
        result.reserve(std::filesystem::file_size(bufferPath));

        std::ifstream input(bufferPath, std::ios::binary);
        result.assign(std::istreambuf_iterator<char>(input), {});

        return result;
    }

    void ImageInfo::Deleter::operator()(unsigned char * data) const
    {
        stbi_image_free(data);
    }

    ImageInfo loadImage(std::filesystem::path const & imagePath)
    {
        ImageInfo result;
        result.data.reset(stbi_load(imagePath.c_str(), &result.width, &result.height, &result.channels, 0));
        if (result.channels == 3)
        {
            // Reload with 4 channels, since WebGPU doesn't support 3-channel textures
            result.data.reset(stbi_load(imagePath.c_str(), &result.width, &result.height, &result.channels, 4));
            result.channels = 4;
        }
        return result;
    }

    ImageInfo loadImage(std::filesystem::path const & assetPath, std::string const & imageUri)
    {
        return loadImage(assetPath.parent_path() / imageUri);
    }

}
