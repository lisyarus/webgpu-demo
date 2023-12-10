#pragma once

#include <webgpu-demo/gltf_asset.hpp>

#include <filesystem>

namespace glTF
{

    Asset load(std::filesystem::path const & path);

    std::vector<char> loadBuffer(std::filesystem::path const & assetPath, std::string const & bufferUri);

    struct ImageInfo
    {
        int width;
        int height;
        int channels;
        std::unique_ptr<unsigned char[]> data;
    };

    ImageInfo loadImage(std::filesystem::path const & assetPath, std::string const & imageUri);

}
