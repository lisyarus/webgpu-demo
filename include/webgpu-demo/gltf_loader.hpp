#pragma once

#include <webgpu-demo/gltf_asset.hpp>

#include <filesystem>

namespace glTF
{

    Asset load(std::filesystem::path const & path);

    std::vector<char> loadBuffer(std::filesystem::path const & assetPath, std::string const & bufferUri);

    struct ImageInfo
    {
        struct Deleter
        {
            void operator()(unsigned char *) const;
        };

        int width;
        int height;
        int channels;
        std::unique_ptr<unsigned char, Deleter> data;
    };

    ImageInfo loadImage(std::filesystem::path const & imagePath);
    ImageInfo loadImage(std::filesystem::path const & assetPath, std::string const & imageUri);

}
