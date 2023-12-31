#include <webgpu-demo/engine_utils.hpp>
#include <webgpu-demo/gltf_loader.hpp>

#include <vector>
#include <fstream>

glm::mat4 glToVkProjection(glm::mat4 matrix)
{
    // Map from [-1, 1] z-range in OpenGL to [0, 1] z-range in Vulkan, WebGPU, etc
    // Equivalent to doing v.z = (v.z + v.w) / 2 after applying the matrix

    for (int i = 0; i < 4; ++i)
        matrix[i][2] = (matrix[i][2] + matrix[i][3]) / 2.f;

    return matrix;
}

const char mainShader[] =
R"(

struct Camera {
    viewProjection : mat4x4f,
    viewProjectionInverse : mat4x4f,
    position : vec3f,
    shock : vec4f,
    shockDirection : vec3f,
    time : f32,
}

struct Object {
    model : mat4x4f,
    baseColorFactor : vec4f,
    metallicFactor : f32,
    roughnessFactor : f32,
    emissiveFactor : vec3f,
}

struct Lights {
    shadowProjection : mat4x4f,
    ambientLight : vec3f,
    sunDirection : vec3f,
    sunIntensity : vec3f,
    envIntensity : f32,
    pointLightCount : u32,
}

struct PointLight {
    position : vec3f,
    intensity : vec3f,
}

struct Water {
    cellSize : vec2f,
    dt : f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> object: Object;

@group(2) @binding(0) var textureSampler: sampler;
@group(2) @binding(1) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(2) var normalTexture: texture_2d<f32>;
@group(2) @binding(3) var metallicRoughnessTexture: texture_2d<f32>;

@group(3) @binding(0) var<uniform> lights : Lights;
@group(3) @binding(1) var shadowSampler: sampler;
@group(3) @binding(2) var shadowMapTexture: texture_2d<f32>;
@group(3) @binding(3) var envSampler: sampler;
@group(3) @binding(4) var envMapTexture : texture_2d<f32>;
@group(3) @binding(5) var<storage, read> pointLights : array<PointLight>;
@group(3) @binding(6) var noise3DTexture : texture_3d<f32>;
@group(3) @binding(7) var noise3DSampler : sampler;

@group(4) @binding(0) var hdrColor : texture_2d<f32>;
@group(4) @binding(1) var hdrDepth : texture_depth_multisampled_2d;
@group(4) @binding(2) var waterData : texture_2d<f32>;
@group(4) @binding(3) var<uniform> water : Water;

struct VertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @location(2) tangent: vec4f,
    @location(3) texcoord : vec2f,
    @location(4) rotation : vec4f,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPosition : vec3f,
    @location(1) normal : vec3f,
    @location(2) tangent : vec4f,
    @location(3) texcoord : vec2f,
}

fn asMat3x3(m : mat4x4f) -> mat3x3f {
    return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}

fn quatMult(q1 : vec4f, q2 : vec4f) -> vec4f {
    return vec4f(q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz), q1.w * q2.w - dot(q1.xyz, q2.xyz));
}

fn quatRotate(q : vec4f, v : vec3f) -> vec3f {
    return quatMult(quatMult(q, vec4f(v, 0.0)), vec4f(-q.xyz, q.w)).xyz;
}

@vertex
fn vertexMain(in : VertexInput) -> VertexOutput {
    let worldPosition = (object.model * vec4f(in.position, 1.0)).xyz;
    let position : vec4f = camera.viewProjection * vec4f(worldPosition, 1.0);
    let normal : vec3f = normalize(asMat3x3(object.model) * quatRotate(in.rotation, in.normal));
    let tangent : vec4f = vec4f(normalize(asMat3x3(object.model) * quatRotate(in.rotation, in.tangent.xyz)), in.tangent.w);
    return VertexOutput(position, worldPosition, normal, tangent, in.texcoord);
}

fn perspectiveDivide(p : vec4f) -> vec3f {
    return p.xyz / p.w;
}

const PI = 3.141592653589793;

fn envDirectionToTexcoord(dir : vec3f) -> vec2f {
    let x = atan2(dir.z, dir.x) / PI * 0.5 + 0.5;
    let y = acos(dir.y) / PI;
    return vec2f(x, y);
}

fn sampleEnvMap(dir : vec3f) -> vec3f {
    // Help the GPU with mipmap selection
    // on cylindrical X coordinate border

    let texcoord1 = envDirectionToTexcoord(dir);
    let texcoord2 = vec2f(modf(texcoord1.x + 0.5).fract - 0.5, texcoord1.y);

    let sample1 = textureSample(envMapTexture, envSampler, texcoord1).rgb * lights.envIntensity;
    let sample2 = textureSample(envMapTexture, envSampler, texcoord2).rgb * lights.envIntensity;

    if (abs(texcoord1.x - 0.5) < 0.25) {
        return sample1;
    } else {
        return sample2;
    }
}

fn specularD(halfway : vec3f, normal : vec3f, alpha2 : f32) -> f32 {
    let ndoth = dot(halfway, normal);
    let denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0);
    return alpha2 * step(0.0, ndoth) / PI / denom / denom;
}

fn specularVHelper(halfway : vec3f, normal : vec3f, alpha2 : f32, v : vec3f) -> f32 {
    let hdotv = dot(halfway, v);
    let ndotv = dot(normal, v);

    return step(0.0, hdotv) / (abs(ndotv) + sqrt(mix(ndotv * ndotv, 1.0, alpha2)));
}

// The BRDF is implemented as described in glTF 2.0 specification:
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
fn computeLighting(
    normal : vec3f,
    baseColor : vec3f,
    metallic : f32,
    roughness : f32,
    viewDirection : vec3f,
    lightDirection : vec3f,
    lightIntensity : vec3f
) -> vec3f
{
    let halfway = normalize(lightDirection + viewDirection);
    let reflected = reflect(-viewDirection, normal);

    let fresnelFactor = pow(1.0 - abs(dot(viewDirection, halfway)), 5.0);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    let diffuse = (1.0 / PI) * baseColor;
    let specular = specularVHelper(halfway, normal, alpha2, lightDirection)
        * specularVHelper(halfway, normal, alpha2, viewDirection)
        * specularD(halfway, normal, alpha2);
    let dielectric = mix(diffuse, vec3f(specular), mix(fresnelFactor, 1.0, 0.04));
    let metal = specular * mix(vec3f(fresnelFactor), vec3f(1.0), baseColor);
    let material = mix(dielectric, metal, metallic);

    let lightness = max(0.0, dot(normal, lightDirection));

    return material * lightness * lightIntensity;
}

fn fractalNoise(p : vec3f) -> f32
{
    return textureSample(noise3DTexture, noise3DSampler, p).r;
}

const PHI = 1.618033988749;

fn volumetricLight(
    lightPosition : vec3f,
    lightIntensity : vec3f,
    rayStart : vec3f,
    rayEnd : vec3f,
    radius : f32
) -> vec3f
{
    let d = normalize(rayEnd - rayStart);
    let r = rayStart - lightPosition;
    let maxDist = length(rayEnd - rayStart);

    // |s + t * d - p|^2 = r^2
    // (s-p)^2 + 2t(s-p)d + t^2 d^2 - r^2 = 0

    let rdotd = dot(r, d);

    let D = rdotd * rdotd - (dot(r, r) - radius * radius);
    if (D <= 0.0) {
        return vec3f(0.0);
    }

    let tmin = clamp(- rdotd - sqrt(D), 0.0, maxDist);
    let tmax = clamp(- rdotd + sqrt(D), 0.0, maxDist);
    let N = 32;
    let dt = (tmax - tmin) / f32(N);
    var result = 0.0;
    for (var i = 0; i < N; i += 1) {
        let p = rayStart + (tmin + (f32(i) + 0.5) * dt) * d;
        var v =
              0.5 * fractalNoise(p * vec3f(1.0, 0.5, 1.0) * 5.0       + vec3f(0.0, - camera.time * 2.0, 0.0))
            + 0.5 * fractalNoise(p * vec3f(1.0, 0.5, 1.0) * 5.0 * PHI + vec3f(0.0, - camera.time * 2.0 / PHI, 0.0));

        v *= pow(1.0 - length(p.xz - lightPosition.xz) / radius, 0.5);
        v -= mix(0.3, 0.7, pow(0.5 + 0.5 * (p.y - lightPosition.y) / radius, 2.5));
        v = max(0.0, 4.0 * v);

        result += v * dt;
    }

    return lightIntensity * result;
}

fn randomizeLightPosition(p : vec3f) -> vec3f
{
    let t = 0.1 * camera.time;
    let s = 0.05;

    var q = p;
    q.x += s * (-1.0 + 2.0 * fractalNoise(p + vec3f(4.25 * t, 0.0, 0.0)));
    q.y += s * (-1.0 + 2.0 * fractalNoise(p + vec3f(0.0, 2.57 * t, 0.0)));
    q.z += s * (-1.0 + 2.0 * fractalNoise(p + vec3f(0.0, 0.0, 1.26 * t)));

    return q;
}

const ESM_FACTOR = 80.0;

@fragment
fn fragmentMain(in : VertexOutput, @builtin(front_facing) front_facing : bool) -> @location(0) vec4f {
    let baseColorSample = textureSample(baseColorTexture, textureSampler, in.texcoord) * object.baseColorFactor;

    let baseColor = baseColorSample.rgb;

    var tbn = mat3x3f();
    tbn[2] = normalize(in.normal);
    tbn[0] = normalize(in.tangent.xyz - tbn[2] * dot(in.tangent.xyz, tbn[2]));
    tbn[1] = cross(tbn[2], tbn[0]) * in.tangent.w;

    let normal = tbn * normalize(2.0 * textureSample(normalTexture, textureSampler, in.texcoord).rgb - vec3(1.0)) * select(-1.0, 1.0, front_facing);

    let materialSample = textureSample(metallicRoughnessTexture, textureSampler, in.texcoord);

    let metallic = materialSample.b * object.metallicFactor;
    let roughness = materialSample.g * object.roughnessFactor;

    let viewDirection = normalize(camera.position - in.worldPosition);

    var litColor = lights.ambientLight * baseColor;

    // Sun contribution

    let shadowPositionClip = lights.shadowProjection * vec4(in.worldPosition, 1.0);
    let shadowPositionNdc = perspectiveDivide(shadowPositionClip);
    let shadowThreshold = 0.125;

    let shadowSample = textureSample(shadowMapTexture, shadowSampler, shadowPositionNdc.xy * vec2f(0.5, -0.5) + vec2f(0.5)).r;
    let shadowFactor = clamp((exp(- ESM_FACTOR * shadowPositionNdc.z) * shadowSample - shadowThreshold) / (1.0 - shadowThreshold), 0.0, 1.0);

    litColor += computeLighting(normal, baseColor, metallic, roughness, viewDirection, lights.sunDirection, shadowFactor * lights.sunIntensity);

    for (var i = 0u; i < lights.pointLightCount; i += 1u) {
        let light = pointLights[i];
        let position = randomizeLightPosition(light.position);
        let direction = position - in.worldPosition;
        let distance = length(direction);
        let radius = 0.1;
        let attenuation = 1.0 / pow(1.0 + distance / radius, 2.0);

        litColor += computeLighting(normal, baseColor, metallic, roughness, viewDirection, direction / distance, light.intensity * attenuation);

        litColor += volumetricLight(light.position, light.intensity, camera.position, in.worldPosition, 2.0 * radius);
    }

    let shockDelta = in.worldPosition - camera.shock.xyz;
    let shockDistance = length(shockDelta);
    let shockD = camera.shock.w - shockDistance;
    let shockA = 0.5 + 0.5 * dot(shockDelta / shockDistance, camera.shockDirection);
    let shockFactor = (exp(- shockD * shockD * 10.0) + 0.25 * step(0.0, shockD) * exp(- shockD * shockD * 0.4) * (0.5 + 0.5 * sin(shockD * 16.0))) * pow(shockA, 256.0);

    let shockColor = mix(litColor, lights.sunIntensity * 0.3, shockFactor);

    return vec4f(shockColor, baseColorSample.a);
}

struct ShadowVertexInput {
    @location(0) position : vec3f,
    @location(1) texcoord : vec2f,
}

struct ShadowVertexOutput {
    @builtin(position) position : vec4f,
    @location(0) texcoord : vec2f,
}

@vertex
fn shadowVertexMain(in : ShadowVertexInput) -> ShadowVertexOutput {
    let position : vec4f = camera.viewProjection * object.model * vec4f(in.position, 1.0);
    return ShadowVertexOutput(position, in.texcoord);
}

@fragment
fn shadowFragmentMain(in : ShadowVertexOutput) -> @location(0) vec4f {
    let baseColor = textureSample(baseColorTexture, textureSampler, in.texcoord) * object.baseColorFactor;

    if (baseColor.a < 0.5) {
        discard;
    }

    return vec4f(exp(ESM_FACTOR * in.position.z), 0.0, 0.0, 0.0);
}

struct EnvVertexInput {
    @builtin(vertex_index) index : u32,
}

struct EnvVertexOutput {
    @builtin(position) position : vec4f,
    @location(0) vertex : vec2f,
}

@vertex
fn envVertexMain(in : EnvVertexInput) -> EnvVertexOutput {
    var vertex : vec2f;
    if (in.index == 0u) {
        vertex = vec2f(-1.0, -1.0);
    } else if (in.index == 1u) {
        vertex = vec2f( 3.0, -1.0);
    } else {
        vertex = vec2f(-1.0,  3.0);
    }

    return EnvVertexOutput(vec4f(vertex, 0.0, 1.0), vertex);
}

@fragment
fn envFragmentMain(in : EnvVertexOutput) -> @location(0) vec4f {
    let p0 = perspectiveDivide(camera.viewProjectionInverse * vec4f(in.vertex, 0.0, 1.0));
    let p1 = perspectiveDivide(camera.viewProjectionInverse * vec4f(in.vertex, 1.0, 1.0));
    let direction = normalize(p1 - p0);

    let sunCircle = smoothstep(0.9999, 0.99999, dot(direction, lights.sunDirection)) * lights.sunIntensity;

    return vec4f(sampleEnvMap(direction) + sunCircle, 1.0);
}

struct WaterVertexInput {
    @builtin(vertex_index) index : u32,
    @location(0) position : vec2f,
}

struct WaterVertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPosition : vec3f,
    @location(1) normal : vec3f,
}

fn waterTriangleNormal(h : f32, p1 : vec3f, p2 : vec3f) -> vec3f
{
    let scale = vec3f(water.cellSize, 1.0);

    let p = vec3f(0.0, 0.0, h);

    let n = normalize(cross(p1 * scale - p, p2 * scale - p));

    return n.xzy;
}

@vertex
fn waterVertexMain(in : WaterVertexInput) -> WaterVertexOutput {
    let count = textureDimensions(waterData, 0);
    let idx = vec2u(in.index % count.x, in.index / count.x);
    let height = textureLoad(waterData, idx, 0).r;

    let hasNX = idx.x > 0u;
    let hasPX = idx.x + 1u < count.x;
    let hasNY = idx.y > 0u;
    let hasPY = idx.y + 1u < count.y;

    let idxNX = max(idx.x, 1u) - 1u;
    let idxPX = min(idx.x, count.x - 2u) + 1u;
    let idxNY = max(idx.y, 1u) - 1u;
    let idxPY = min(idx.y, count.y - 2u) + 1u;

    let heightNX = textureLoad(waterData, vec2u(idxNX, idx.y), 0).r;
    let heightPX = textureLoad(waterData, vec2u(idxPX, idx.y), 0).r;
    let heightNY = textureLoad(waterData, vec2u(idx.x, idxNY), 0).r;
    let heightPY = textureLoad(waterData, vec2u(idx.x, idxPY), 0).r;

    var sumNormals = vec3f(0.0);

    if (hasNX && hasNY) {
        sumNormals += waterTriangleNormal(height, vec3f(-1.0, 0.0, heightNX), vec3f(0.0, -1.0, heightNY));
    }

    if (hasPX && hasNY) {
        sumNormals += waterTriangleNormal(height, vec3f(0.0, -1.0, heightNY), vec3f(1.0, 0.0, heightPX));
    }

    if (hasNX && hasPY) {
        sumNormals += waterTriangleNormal(height, vec3f(0.0, 1.0, heightPY), vec3f(-1.0, 0.0, heightNX));
    }

    if (hasPX && hasPY) {
        sumNormals += waterTriangleNormal(height, vec3f(1.0, 0.0, heightPX), vec3f(0.0, 1.0, heightPY));
    }

    let worldPosition = vec3f(in.position.x, 0.5 + height, in.position.y);
    let normal = normalize(sumNormals);
    let position = camera.viewProjection * vec4f(worldPosition, 1.0);
    return WaterVertexOutput(position, worldPosition, normal);
}

fn waterSpecular(normal : vec3f, viewDirection : vec3f, lightDirection : vec3f, lightIntensity : vec3f) -> vec3f
{
    let reflectivity = mix(0.02, 1.0, pow(1.0 - dot(viewDirection, normal), 5.0));

    let lightness = max(0.0, dot(normal, lightDirection));
    let halfway = normalize(viewDirection + lightDirection);
    let specular = pow(max(0.0, dot(halfway, normal)), 256.0);

    return lightIntensity * lightness * specular * reflectivity * 64.0;
}

fn refract(normal : vec3f, to_camera : vec3f, n : f32) -> vec3f
{
    let cosI = abs(dot(normal, to_camera));
    let sinT2 = n * n * max(0.0, 1.0 - cosI * cosI);
    let cosT = sqrt(1.0 - sinT2);
    return -n * to_camera + (n * cosI - cosT) * normal;
}

fn remapReflection(in : vec2f, threshold : f32) -> vec2f
{
    let s = threshold;
    return sign(in) * select(mix(2.0 / (1.0 + exp(- 2.0 * (abs(in) - vec2f(s)) / (1.0 - s))) - 1.0, vec2f(1.0), s), abs(in), abs(in) < vec2f(s));
}

@fragment
fn waterFragmentMain(in : WaterVertexOutput) -> @location(0) vec4f {
    let viewportSize = textureDimensions(hdrDepth);
    let rayEndDepth = textureLoad(hdrDepth, vec2i(in.position.xy), 0);

    if (in.position.z > rayEndDepth) {
        discard;
    }

    let rayEndPosition = perspectiveDivide(camera.viewProjectionInverse * vec4f((2.0 * in.position.xy / vec2f(viewportSize) - vec2f(1.0)) * vec2f(1.0, -1.0), rayEndDepth, 1.0));
    let distance = length(in.worldPosition - rayEndPosition);

    let normal = normalize(in.normal);
    let viewDirection = normalize(camera.position - in.worldPosition);

    let refractedDirection = refract(normal, viewDirection, 0.95);
    let refractedPosition = in.worldPosition + distance * refractedDirection;
    let refractedNdc = perspectiveDivide(camera.viewProjection * vec4(refractedPosition, 1.0));
    let refractedFragCoord = vec2i((remapReflection(refractedNdc.xy * vec2f(1.0, -1.0), 0.9) * 0.5 + vec2f(0.5)) * vec2f(viewportSize));
    let refractedEndDepth = textureLoad(hdrDepth, refractedFragCoord, 0);
    let refractedEndPosition = perspectiveDivide(camera.viewProjectionInverse * vec4f((2.0 * in.position.xy / vec2f(viewportSize) - vec2f(1.0)) * vec2f(1.0, -1.0), refractedEndDepth, 1.0));
    let refractedDistance = length(refractedPosition - refractedEndPosition);

    let colorIn = textureLoad(hdrColor, refractedFragCoord, 0).rgb;
    let waterColor = vec3f(2.0);

    let fadeColor = mix(lights.ambientLight * vec3f(0.5, 0.5, 0.8), colorIn * waterColor, exp(- 0.3 * refractedDistance));

    var resultColor = fadeColor;

    let shadowPositionClip = lights.shadowProjection * vec4(in.worldPosition, 1.0);
    let shadowPositionNdc = perspectiveDivide(shadowPositionClip);
    let shadowThreshold = 0.125;

    let shadowSample = textureSample(shadowMapTexture, shadowSampler, shadowPositionNdc.xy * vec2f(0.5, -0.5) + vec2f(0.5)).r;
    let shadowFactor = clamp((exp(- ESM_FACTOR * shadowPositionNdc.z) * shadowSample - shadowThreshold) / (1.0 - shadowThreshold), 0.0, 1.0);

    resultColor += waterSpecular(normal, viewDirection, lights.sunDirection, lights.sunIntensity * shadowFactor);

    for (var i = 0u; i < lights.pointLightCount; i += 1u) {
        let light = pointLights[i];
        let position = randomizeLightPosition(light.position);
        let direction = position - in.worldPosition;
        let distance = length(direction);
        let radius = 0.1;
        let attenuation = 1.0 / pow(1.0 + distance / radius, 2.0);

        resultColor += waterSpecular(normal, viewDirection, direction / distance, light.intensity * attenuation);
    }

    return vec4f(resultColor, 1.0);
}

)";

const char genMipmapShader[] =
R"(

@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<rgba8unorm, write>;

const GAMMA : f32 = 2.2;

fn fromSRGB(c : vec4f) -> vec4f {
    return vec4f(pow(c.rgb, vec3f(GAMMA)), c.a);
}

fn toSRGB(c : vec4f) -> vec4f {
    return vec4f(pow(c.rgb, vec3f(1.0 / GAMMA)), c.a);
}

fn premult(c : vec4f) -> vec4f {
    return vec4f(c.rgb * c.a, c.a);
}

fn unpremult(c : vec4f) -> vec4f {
    if (c.a == 0.0) {
        return c;
    }
    return vec4f(c.rgb / c.a, c.a);
}

@compute @workgroup_size(8, 8)
fn generateMipmap(@builtin(global_invocation_id) id : vec3<u32>) {
    if (all(id.xy < textureDimensions(output))) {
        let sum =
              premult(textureLoad(input, 2u * id.xy + vec2u(0u, 0u), 0))
            + premult(textureLoad(input, 2u * id.xy + vec2u(0u, 1u), 0))
            + premult(textureLoad(input, 2u * id.xy + vec2u(1u, 0u), 0))
            + premult(textureLoad(input, 2u * id.xy + vec2u(1u, 1u), 0))
            ;

        let result = unpremult(sum / 4.0);

        textureStore(output, id.xy, result);
    }
}

@compute @workgroup_size(8, 8)
fn generateMipmapSRGB(@builtin(global_invocation_id) id : vec3<u32>) {
    if (all(id.xy < textureDimensions(output))) {
        let sum =
              premult(fromSRGB(textureLoad(input, 2u * id.xy + vec2u(0u, 0u), 0)))
            + premult(fromSRGB(textureLoad(input, 2u * id.xy + vec2u(0u, 1u), 0)))
            + premult(fromSRGB(textureLoad(input, 2u * id.xy + vec2u(1u, 0u), 0)))
            + premult(fromSRGB(textureLoad(input, 2u * id.xy + vec2u(1u, 1u), 0)))
            ;

        let result = toSRGB(unpremult(sum / 4.0));

        textureStore(output, id.xy, result);
    }
}

)";

const char genEnvMipmapShader[] =
R"(

@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn generateMipmapEnv(@builtin(global_invocation_id) id : vec3<u32>) {
    if (all(id.xy < textureDimensions(output))) {
        let sum =
              textureLoad(input, 2u * id.xy + vec2u(0u, 0u), 0)
            + textureLoad(input, 2u * id.xy + vec2u(0u, 1u), 0)
            + textureLoad(input, 2u * id.xy + vec2u(1u, 0u), 0)
            + textureLoad(input, 2u * id.xy + vec2u(1u, 1u), 0)
            ;

        let result = sum / 4.0;

        textureStore(output, id.xy, result);
    }
}

)";

const char blurShadowShader[] =
R"(

@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<r32float, write>;

const RADIUS = 7;
const SIGMA = 5.0;

fn doBlur(id : vec2u, direction : vec2i) {
    var sum = 0.0;
    var sumWeights = 0.0;
    for (var i = -RADIUS; i <= RADIUS; i += 1) {
        let weight = exp(- f32(i * i) / (SIGMA * SIGMA));
        sum += textureLoad(input, vec2u(vec2i(id) + direction * i), 0).r * weight;
        sumWeights += weight;
    }

    sum /= sumWeights;

    textureStore(output, id, vec4f(sum, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(32, 1)
fn blurShadowX(@builtin(global_invocation_id) id : vec3u) {
    doBlur(id.xy, vec2i(1, 0));
}

@compute @workgroup_size(1, 32)
fn blurShadowY(@builtin(global_invocation_id) id : vec3u) {
    doBlur(id.xy, vec2i(0, 1));
}

)";

const char simulateClothShader[] =
R"(

struct Vertex {
    // Bypass alignment issues
    data : array<f32, 16>,
}

struct ClothVertex {
    oldVelocity : vec3f,
    velocity : vec3f,
    newPosition : vec3f,
}

struct ClothEdge {
    delta : vec4f,
    id : u32,
}

struct ClothSettings {
    dt : f32,
    gravity : f32,
}

struct Camera {
    viewProjection : mat4x4f,
    viewProjectionInverse : mat4x4f,
    position : vec3f,
    shock : vec4f,
    shockDirection : vec3f,
    time : f32,
}

@group(0) @binding(0) var<storage, read_write> vertices : array<Vertex>;
@group(0) @binding(1) var<storage, read_write> clothVertices : array<ClothVertex>;
@group(0) @binding(2) var<storage, read> clothEdges : array<ClothEdge>;
@group(0) @binding(3) var<uniform> settings : ClothSettings;

@group(1) @binding(0) var<uniform> camera: Camera;

const CLOTH_EDGES_PER_VERTEX = 8u;
const SPRING_FORCE = 20000.0;
const MASS = 0.1;
const DAMPING = 0.04;
const WATER_DAMPING = 4.0;
const SMOOTHING = 10.0;
const FRICTION = 1.0;
const SHOCK = 20.0;

fn getPosition(id : u32) -> vec3f {
    return vec3f(vertices[id].data[0], vertices[id].data[1], vertices[id].data[2]);
}

fn setPosition(id : u32, value : vec3f) {
    vertices[id].data[0] = value.x;
    vertices[id].data[1] = value.y;
    vertices[id].data[2] = value.z;
}

fn getRotation(id : u32) -> vec4f {
    return vec4f(vertices[id].data[12], vertices[id].data[13], vertices[id].data[14], vertices[id].data[15]);
}

fn setRotation(id : u32, value : vec4f) {
    vertices[id].data[12] = value.x;
    vertices[id].data[13] = value.y;
    vertices[id].data[14] = value.z;
    vertices[id].data[15] = value.w;
}

struct CollideResult {
    position : vec3f,
    velocity : vec3f,
}

fn collideCamera(position : vec3f, velocity : vec3f, radius : f32) -> CollideResult {
    let delta = position - camera.position;
    let distance = length(delta);
    if (distance >= radius) {
        return CollideResult(position, velocity);
    } else {
        let n = delta / distance;
        let newPosition = camera.position + n * radius;
        let newVelocity = (velocity - n * dot(velocity, n)) * exp(- FRICTION * settings.dt);
        return CollideResult(newPosition, newVelocity);
    }
}

@compute @workgroup_size(32)
fn simulateCloth(@builtin(global_invocation_id) id : vec3u) {
    let currentPosition = getPosition(id.x);

    var force = vec3f(0.0);
    var avgVelocity = vec3f(0.0);
    var edgeCount = 0;
    for (var i = 0u; i < CLOTH_EDGES_PER_VERTEX; i += 1u) {
        let edge = clothEdges[id.x * CLOTH_EDGES_PER_VERTEX + i];
        if (edge.id != 0xffffffffu) {
            let delta = getPosition(edge.id) - currentPosition;
            let distance = length(delta);
            force += delta * (SPRING_FORCE * (distance - edge.delta.w) / distance);

            avgVelocity += clothVertices[edge.id].oldVelocity;

            edgeCount += 1;
        }
    }

    if (edgeCount > 0) {
        force += vec3f(0.0, - settings.gravity * MASS, 0.0);

        let shockDelta = currentPosition - camera.shock.xyz;
        let shockDistance = length(shockDelta);
        let shockDir = shockDelta / shockDistance;
        let shockD = camera.shock.w - shockDistance;
        let shockA = 0.5 + 0.5 * dot(shockDir, camera.shockDirection);
        let shockStrength = SHOCK * exp(- shockD * shockD * 1.0) * pow(shockA, 256.0);

        force += shockDir * shockStrength;

        let currentVelocity = clothVertices[id.x].velocity;

        avgVelocity /= f32(edgeCount);

//        let damping = select(DAMPING, WATER_DAMPING, currentPosition.y < 0.5);
        let damping = DAMPING;

        let newVelocity = (currentVelocity + force * settings.dt / MASS) * exp(- damping * settings.dt);
        let smoothedVelocity = mix(avgVelocity, newVelocity, exp(- SMOOTHING * settings.dt));

        let collision = collideCamera(currentPosition + smoothedVelocity * settings.dt, smoothedVelocity, 0.5);

        clothVertices[id.x].velocity = collision.velocity;
        clothVertices[id.x].newPosition = collision.position;
    } else {
        clothVertices[id.x].newPosition = currentPosition;
    }
}

// The matrix of the linear operator (q -> q * a)
fn rightMultMatrix(a : vec4f) -> mat4x4f {
    return mat4x4f(
        vec4f( a.w, -a.z,  a.y, -a.x),
        vec4f( a.z,  a.w, -a.x, -a.y),
        vec4f(-a.y,  a.x,  a.w, -a.z),
        vec4f( a.x,  a.y,  a.z,  a.w),
    );
}

// The matrix of the linear operator (q -> b * q)
fn leftMultMatrix(b : vec4f) -> mat4x4f {
    return mat4x4f(
        vec4f( b.w,  b.z, -b.y, -b.x),
        vec4f(-b.z,  b.w,  b.x, -b.y),
        vec4f( b.y, -b.x,  b.w, -b.z),
        vec4f( b.x,  b.y,  b.z,  b.w),
    );
}

@compute @workgroup_size(32)
fn simulateClothCopy(@builtin(global_invocation_id) id : vec3u) {
    var errorMatrix = mat4x4f(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));

    let currentPosition = clothVertices[id.x].newPosition;

    for (var i = 0u; i < CLOTH_EDGES_PER_VERTEX; i += 1u) {
        let edge = clothEdges[id.x * CLOTH_EDGES_PER_VERTEX + i];
        if (edge.id != 0xffffffffu) {
            let delta = edge.delta.xyz;
            let currentDelta = clothVertices[edge.id].newPosition - currentPosition;

            let m = rightMultMatrix(vec4f(delta, 0.0)) - leftMultMatrix(vec4f(currentDelta, 0.0));
            errorMatrix += transpose(m) * m;
        }
    }

    let currentRotation = getRotation(id.x);
    let rotationGrad = 2.0 * currentRotation * errorMatrix;
    let newRotation = normalize(currentRotation - rotationGrad * 0.25);
    setRotation(id.x, newRotation);

    clothVertices[id.x].oldVelocity = clothVertices[id.x].velocity;
    setPosition(id.x, clothVertices[id.x].newPosition);
}

)";

const char simulateWaterShader[] =
R"(

struct Water
{
    cellSize : vec2f,
    dt : f32,
}

@group(0) @binding(0) var<uniform> water : Water;
@group(0) @binding(1) var oldWaterData : texture_2d<f32>;
@group(0) @binding(2) var newWaterData : texture_storage_2d<rg32float, write>;

const DX = 1.0;
const C = 10.0;
const DAMPING = 0.0;

@compute @workgroup_size(8, 8)
fn simulateWater(@builtin(global_invocation_id) id : vec3u) {
    let oldData = textureLoad(oldWaterData, id.xy, 0).rg;
    let count = textureDimensions(oldWaterData, 0);

    let hasNX = id.x > 0u;
    let hasPX = id.x + 1u < count.x;
    let hasNY = id.y > 0u;
    let hasPY = id.y + 1u < count.y;

    let idNX = max(id.x, 1u) - 1u;
    let idPX = min(id.x, count.x - 2u) + 1u;
    let idNY = max(id.y, 1u) - 1u;
    let idPY = min(id.y, count.y - 2u) + 1u;

    let heightNX = textureLoad(oldWaterData, vec2u(idNX, id.y), 0).r;
    let heightPX = textureLoad(oldWaterData, vec2u(idPX, id.y), 0).r;
    let heightNY = textureLoad(oldWaterData, vec2u(id.x, idNY), 0).r;
    let heightPY = textureLoad(oldWaterData, vec2u(id.x, idPY), 0).r;

    var L = 0.0;
    if (hasNX) { L += heightNX - oldData.r; }
    if (hasPX) { L += heightPX - oldData.r; }
    if (hasNY) { L += heightNY - oldData.r; }
    if (hasPY) { L += heightPY - oldData.r; }

    var newData = vec2f(0.0);
    newData.g = oldData.g + C * C * L / DX / DX * water.dt;
    newData.g *= exp(- DAMPING * water.dt);
    newData.r = oldData.r + newData.g * water.dt;

    textureStore(newWaterData, id.xy, vec4f(newData, 0.0, 0.0));
}

)";

const char ldrShader[] =
R"(

@group(0) @binding(0) var hdrTexture : texture_2d<f32>;

@vertex
fn ldrVertexMain(@builtin(vertex_index) index : u32) -> @builtin(position) vec4f {
    if (index == 0u) {
        return vec4f(-1.0, -1.0, 0.0, 1.0);
    }
    else if (index == 1u) {
        return vec4f( 3.0, -1.0, 0.0, 1.0);
    }
    else {
        return vec4f(-1.0,  3.0, 0.0, 1.0);
    }
}

fn Uncharted2TonemapImpl(x : vec3f) -> vec3f {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

fn ACESFilm(x : vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

fn tonemap(x : vec3f) -> vec3f {
    return Uncharted2TonemapImpl(x) / Uncharted2TonemapImpl(vec3f(3.0));
//    return ACESFilm(x);
}

@fragment
fn ldrFragmentMain(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
    let color = textureLoad(hdrTexture, vec2u(fragCoord.xy), 0).rgb;
    return vec4f(tonemap(color), 1.0);
}

)";

const char textShader[] =
R"(

@group(0) @binding(0) var<uniform> viewportSize : vec2i;
@group(0) @binding(1) var fontTexture : texture_2d<f32>;
@group(0) @binding(2) var fontSampler : sampler;

struct VertexIn
{
    @location(0) position : vec2f,
    @location(1) texcoord : vec2f,
}

struct VertexOut
{
    @builtin(position) position : vec4f,
    @location(0) texcoord : vec2f,
}

@vertex
fn textVertexMain(in : VertexIn) -> VertexOut {
    return VertexOut(
        vec4f(vec2f(1.0, -1.0) * (2.0 * in.position / vec2f(viewportSize) - vec2f(1.f)), 0.0, 1.0),
        in.texcoord
    );
}

@fragment
fn textFragmentMain(in : VertexOut) -> @location(0) vec4f {
    let alpha = textureSample(fontTexture, fontSampler, in.texcoord).r;
    return vec4f(vec3f(1.0), alpha);
}

)";

WGPUShaderModule createShaderModule(WGPUDevice device, char const * code)
{
    WGPUShaderModuleWGSLDescriptor shaderModuleWGSLDescriptor;
    shaderModuleWGSLDescriptor.chain.next = nullptr;
    shaderModuleWGSLDescriptor.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderModuleWGSLDescriptor.code = code;

    WGPUShaderModuleDescriptor shaderModuleDescriptor;
    shaderModuleDescriptor.nextInChain = &shaderModuleWGSLDescriptor.chain;
    shaderModuleDescriptor.label = nullptr;
    shaderModuleDescriptor.hintCount = 0;
    shaderModuleDescriptor.hints = nullptr;

    return wgpuDeviceCreateShaderModule(device, &shaderModuleDescriptor);
}

std::uint32_t minStorageBufferOffsetAlignment(WGPUDevice device)
{
    WGPUSupportedLimits limits;
    limits.nextInChain = nullptr;
    wgpuDeviceGetLimits(device, &limits);
    return limits.limits.minStorageBufferOffsetAlignment;
}

WGPUBindGroupLayout createEmptyBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 0;
    descriptor.entries = nullptr;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createCameraBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[1];
    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = sizeof(CameraUniform);
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 1;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createObjectBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[1];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = true;
    entries[0].buffer.minBindingSize = sizeof(ObjectUniform);
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 1;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createTexturesBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[4];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_Undefined;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Fragment;
    entries[3].buffer.nextInChain = nullptr;
    entries[3].buffer.type = WGPUBufferBindingType_Undefined;
    entries[3].buffer.hasDynamicOffset = false;
    entries[3].buffer.minBindingSize = 0;
    entries[3].sampler.nextInChain = nullptr;
    entries[3].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[3].texture.nextInChain = nullptr;
    entries[3].texture.sampleType = WGPUTextureSampleType_Float;
    entries[3].texture.multisampled = false;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[3].storageTexture.nextInChain = nullptr;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[3].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 4;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createLightsBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[8];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = sizeof(LightsUniform);
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_Undefined;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Fragment;
    entries[3].buffer.nextInChain = nullptr;
    entries[3].buffer.type = WGPUBufferBindingType_Undefined;
    entries[3].buffer.hasDynamicOffset = false;
    entries[3].buffer.minBindingSize = 0;
    entries[3].sampler.nextInChain = nullptr;
    entries[3].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[3].texture.nextInChain = nullptr;
    entries[3].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[3].texture.multisampled = false;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[3].storageTexture.nextInChain = nullptr;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[3].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[4].nextInChain = nullptr;
    entries[4].binding = 4;
    entries[4].visibility = WGPUShaderStage_Fragment;
    entries[4].buffer.nextInChain = nullptr;
    entries[4].buffer.type = WGPUBufferBindingType_Undefined;
    entries[4].buffer.hasDynamicOffset = false;
    entries[4].buffer.minBindingSize = 0;
    entries[4].sampler.nextInChain = nullptr;
    entries[4].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[4].texture.nextInChain = nullptr;
    entries[4].texture.sampleType = WGPUTextureSampleType_Float;
    entries[4].texture.multisampled = false;
    entries[4].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[4].storageTexture.nextInChain = nullptr;
    entries[4].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[4].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[4].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[5].nextInChain = nullptr;
    entries[5].binding = 5;
    entries[5].visibility = WGPUShaderStage_Fragment;
    entries[5].buffer.nextInChain = nullptr;
    entries[5].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[5].buffer.hasDynamicOffset = false;
    entries[5].buffer.minBindingSize = sizeof(PointLight);
    entries[5].sampler.nextInChain = nullptr;
    entries[5].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[5].texture.nextInChain = nullptr;
    entries[5].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[5].texture.multisampled = false;
    entries[5].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[5].storageTexture.nextInChain = nullptr;
    entries[5].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[5].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[5].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[6].nextInChain = nullptr;
    entries[6].binding = 6;
    entries[6].visibility = WGPUShaderStage_Fragment;
    entries[6].buffer.nextInChain = nullptr;
    entries[6].buffer.type = WGPUBufferBindingType_Undefined;
    entries[6].buffer.hasDynamicOffset = false;
    entries[6].buffer.minBindingSize = 0;
    entries[6].sampler.nextInChain = nullptr;
    entries[6].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[6].texture.nextInChain = nullptr;
    entries[6].texture.sampleType = WGPUTextureSampleType_Float;
    entries[6].texture.multisampled = false;
    entries[6].texture.viewDimension = WGPUTextureViewDimension_3D;
    entries[6].storageTexture.nextInChain = nullptr;
    entries[6].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[6].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[6].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[7].nextInChain = nullptr;
    entries[7].binding = 7;
    entries[7].visibility = WGPUShaderStage_Fragment;
    entries[7].buffer.nextInChain = nullptr;
    entries[7].buffer.type = WGPUBufferBindingType_Undefined;
    entries[7].buffer.hasDynamicOffset = false;
    entries[7].buffer.minBindingSize = 0;
    entries[7].sampler.nextInChain = nullptr;
    entries[7].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[7].texture.nextInChain = nullptr;
    entries[7].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[7].texture.multisampled = false;
    entries[7].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[7].storageTexture.nextInChain = nullptr;
    entries[7].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[7].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[7].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 8;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createGenMipmapBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[2];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 2;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createGenEnvMipmapBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[2];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format = WGPUTextureFormat_RGBA32Float;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 2;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createBlurShadowBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[2];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format = WGPUTextureFormat_R32Float;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 2;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createSimulateClothBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[4];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Storage;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Storage;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].buffer.nextInChain = nullptr;
    entries[3].buffer.type = WGPUBufferBindingType_Uniform;
    entries[3].buffer.hasDynamicOffset = false;
    entries[3].buffer.minBindingSize = sizeof(ClothSettingsUniform);
    entries[3].sampler.nextInChain = nullptr;
    entries[3].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[3].texture.nextInChain = nullptr;
    entries[3].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[3].texture.multisampled = false;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[3].storageTexture.nextInChain = nullptr;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[3].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 4;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createWaterBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[4];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Depth;
    entries[1].texture.multisampled = true;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Vertex;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_Undefined;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Vertex;
    entries[3].buffer.nextInChain = nullptr;
    entries[3].buffer.type = WGPUBufferBindingType_Uniform;
    entries[3].buffer.hasDynamicOffset = false;
    entries[3].buffer.minBindingSize = sizeof(WaterUniform);
    entries[3].sampler.nextInChain = nullptr;
    entries[3].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[3].texture.nextInChain = nullptr;
    entries[3].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[3].texture.multisampled = false;
    entries[3].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[3].storageTexture.nextInChain = nullptr;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[3].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 4;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createSimulateWaterBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[3];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = sizeof(WaterUniform);
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Compute;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_Undefined;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[2].storageTexture.format = WGPUTextureFormat_RG32Float;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 3;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createHDRBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[1];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Undefined;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = 0;
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 1;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUBindGroupLayout createTextBindGroupLayout(WGPUDevice device)
{
    WGPUBindGroupLayoutEntry entries[3];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Vertex;
    entries[0].buffer.nextInChain = nullptr;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[0].buffer.hasDynamicOffset = false;
    entries[0].buffer.minBindingSize = sizeof(TextUniform);
    entries[0].sampler.nextInChain = nullptr;
    entries[0].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[0].texture.nextInChain = nullptr;
    entries[0].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[0].texture.multisampled = false;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[0].storageTexture.nextInChain = nullptr;
    entries[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[0].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].buffer.nextInChain = nullptr;
    entries[1].buffer.type = WGPUBufferBindingType_Undefined;
    entries[1].buffer.hasDynamicOffset = false;
    entries[1].buffer.minBindingSize = 0;
    entries[1].sampler.nextInChain = nullptr;
    entries[1].sampler.type = WGPUSamplerBindingType_Undefined;
    entries[1].texture.nextInChain = nullptr;
    entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    entries[1].texture.multisampled = false;
    entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].storageTexture.nextInChain = nullptr;
    entries[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[1].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].buffer.nextInChain = nullptr;
    entries[2].buffer.type = WGPUBufferBindingType_Undefined;
    entries[2].buffer.hasDynamicOffset = false;
    entries[2].buffer.minBindingSize = 0;
    entries[2].sampler.nextInChain = nullptr;
    entries[2].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[2].texture.nextInChain = nullptr;
    entries[2].texture.sampleType = WGPUTextureSampleType_Undefined;
    entries[2].texture.multisampled = false;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_Undefined;
    entries[2].storageTexture.nextInChain = nullptr;
    entries[2].storageTexture.access = WGPUStorageTextureAccess_Undefined;
    entries[2].storageTexture.format = WGPUTextureFormat_Undefined;
    entries[2].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    WGPUBindGroupLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.entryCount = 3;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroupLayout(device, &descriptor);
}

WGPUPipelineLayout createPipelineLayout(WGPUDevice device, std::initializer_list<WGPUBindGroupLayout> bindGroupLayouts)
{
    WGPUPipelineLayoutDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.bindGroupLayoutCount = bindGroupLayouts.size();
    descriptor.bindGroupLayouts = &(*bindGroupLayouts.begin());

    return wgpuDeviceCreatePipelineLayout(device, &descriptor);
}

WGPURenderPipeline createMainPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = WGPUTextureFormat_RGBA16Float;
    colorTargetState.blend = nullptr;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPUVertexAttribute attributes[5];
    attributes[0].format = WGPUVertexFormat_Float32x3;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x3;
    attributes[1].offset = 12;
    attributes[1].shaderLocation = 1;
    attributes[2].format = WGPUVertexFormat_Float32x4;
    attributes[2].offset = 24;
    attributes[2].shaderLocation = 2;
    attributes[3].format = WGPUVertexFormat_Float32x2;
    attributes[3].offset = 40;
    attributes[3].shaderLocation = 3;
    attributes[4].format = WGPUVertexFormat_Float32x4;
    attributes[4].offset = 48;
    attributes[4].shaderLocation = 4;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(Vertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 5;
    vertexBufferLayout.attributes = attributes;

    WGPUDepthStencilState depthStencilState;
    depthStencilState.nextInChain = nullptr;
    depthStencilState.format = WGPUTextureFormat_Depth24Plus;
    depthStencilState.depthWriteEnabled = true;
    depthStencilState.depthCompare = WGPUCompareFunction_Less;
    depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilBack.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;
    depthStencilState.depthBias = 0;
    depthStencilState.depthBiasSlopeScale = 0.f;
    depthStencilState.depthBiasClamp = 0.f;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "main";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "vertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexBufferLayout;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = &depthStencilState;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 4;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = true;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPURenderPipeline createShadowPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUColorTargetState colorTargets[1];
    colorTargets[0].nextInChain = nullptr;
    colorTargets[0].format = WGPUTextureFormat_R32Float;
    colorTargets[0].blend = nullptr;
    colorTargets[0].writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "shadowFragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = colorTargets;

    WGPUVertexAttribute attributes[2];
    attributes[0].format = WGPUVertexFormat_Float32x3;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x2;
    attributes[1].offset = 40;
    attributes[1].shaderLocation = 1;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(Vertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 2;
    vertexBufferLayout.attributes = attributes;

    WGPUDepthStencilState depthStencilState;
    depthStencilState.nextInChain = nullptr;
    depthStencilState.format = WGPUTextureFormat_Depth24Plus;
    depthStencilState.depthWriteEnabled = true;
    depthStencilState.depthCompare = WGPUCompareFunction_Less;
    depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
    depthStencilState.stencilBack.failOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilBack.passOp = WGPUStencilOperation_Keep;
    depthStencilState.stencilReadMask = 0;
    depthStencilState.stencilWriteMask = 0;
    depthStencilState.depthBias = 0;
    depthStencilState.depthBiasSlopeScale = 0.f;
    depthStencilState.depthBiasClamp = 0.f;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "shadow";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "shadowVertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexBufferLayout;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = &depthStencilState;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = false;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPURenderPipeline createEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = WGPUTextureFormat_RGBA16Float;
    colorTargetState.blend = nullptr;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "envFragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "env";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "envVertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 0;
    descriptor.vertex.buffers = nullptr;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = nullptr;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 4;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = false;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPUComputePipeline createMipmapPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "mipmap";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "generateMipmap";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createMipmapSRGBPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "mipmapSRGB";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "generateMipmapSRGB";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createMipmapEnvPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "mipmapEnv";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "generateMipmapEnv";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createBlurShadowXPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "blurShadowX";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "blurShadowX";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createBlurShadowYPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "blurShadowY";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "blurShadowY";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createSimulateClothPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "simulateCloth";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "simulateCloth";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPUComputePipeline createSimulateClothCopyPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "simulateClothCopy";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "simulateClothCopy";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPURenderPipeline createRenderWaterPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = WGPUTextureFormat_RGBA16Float;
    colorTargetState.blend = nullptr;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "waterFragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPUVertexAttribute attributes[1];
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(glm::vec2);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 1;
    vertexBufferLayout.attributes = attributes;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "water";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "waterVertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexBufferLayout;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = nullptr;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = false;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPUComputePipeline createSimulateWaterPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule)
{
    WGPUComputePipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "simulateWater";
    descriptor.layout = pipelineLayout;
    descriptor.compute.nextInChain = nullptr;
    descriptor.compute.module = shaderModule;
    descriptor.compute.entryPoint = "simulateWater";
    descriptor.compute.constantCount = 0;
    descriptor.compute.constants = nullptr;

    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

WGPURenderPipeline createLDRPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule, WGPUTextureFormat surfaceFormat)
{
    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = surfaceFormat;
    colorTargetState.blend = nullptr;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "ldrFragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "ldr";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "ldrVertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 0;
    descriptor.vertex.buffers = nullptr;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = nullptr;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = false;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPURenderPipeline createTextPipeline(WGPUDevice device, WGPUPipelineLayout pipelineLayout, WGPUShaderModule shaderModule, WGPUTextureFormat surfaceFormat)
{
    WGPUVertexAttribute attributes[2];
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x2;
    attributes[1].offset = 8;
    attributes[1].shaderLocation = 1;

    WGPUVertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.arrayStride = sizeof(TextVertex);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
    vertexBufferLayout.attributeCount = 2;
    vertexBufferLayout.attributes = attributes;

    WGPUBlendState blendState;
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState colorTargetState;
    colorTargetState.nextInChain = nullptr;
    colorTargetState.format = surfaceFormat;
    colorTargetState.blend = &blendState;
    colorTargetState.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState;
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "textFragmentMain";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    WGPURenderPipelineDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = "text";
    descriptor.layout = pipelineLayout;
    descriptor.nextInChain = nullptr;
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "textVertexMain";
    descriptor.vertex.constantCount = 0;
    descriptor.vertex.constants = nullptr;
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexBufferLayout;
    descriptor.primitive.nextInChain = nullptr;
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    descriptor.primitive.cullMode = WGPUCullMode_None;
    descriptor.depthStencil = nullptr;
    descriptor.multisample.nextInChain = nullptr;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = -1;
    descriptor.multisample.alphaToCoverageEnabled = false;
    descriptor.fragment = &fragmentState;

    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

WGPUBuffer createUniformBuffer(WGPUDevice device, std::uint64_t size)
{
    WGPUBufferDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    descriptor.size = size;
    descriptor.mappedAtCreation = false;

    return wgpuDeviceCreateBuffer(device, &descriptor);
}

WGPUBuffer createStorageBuffer(WGPUDevice device, std::uint64_t size)
{
    WGPUBufferDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
    descriptor.size = size;
    descriptor.mappedAtCreation = false;

    return wgpuDeviceCreateBuffer(device, &descriptor);
}

WGPUBindGroup createEmptyBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout)
{
    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 0;
    descriptor.entries = nullptr;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createCameraBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer)
{
    WGPUBindGroupEntry bindGroupEntry;
    bindGroupEntry.nextInChain = nullptr;
    bindGroupEntry.binding = 0;
    bindGroupEntry.buffer = uniformBuffer;
    bindGroupEntry.offset = 0;
    bindGroupEntry.size = sizeof(CameraUniform);
    bindGroupEntry.sampler = nullptr;
    bindGroupEntry.textureView = nullptr;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 1;
    descriptor.entries = &bindGroupEntry;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createObjectBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer)
{
    WGPUBindGroupEntry entries[1];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = sizeof(ObjectUniform);
    entries[0].sampler = nullptr;
    entries[0].textureView = nullptr;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 1;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createLightsBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer,
    WGPUSampler shadowSampler, WGPUTextureView shadowMapView, WGPUSampler envSampler, WGPUTextureView envMapView,
    WGPUBuffer pointLightsBuffer, WGPUTextureView noise3DView, WGPUSampler noise3DSampler)
{
    WGPUBindGroupEntry entries[8];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = sizeof(LightsUniform);
    entries[0].sampler = nullptr;
    entries[0].textureView = nullptr;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = nullptr;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = shadowSampler;
    entries[1].textureView = nullptr;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].buffer = nullptr;
    entries[2].offset = 0;
    entries[2].size = 0;
    entries[2].sampler = nullptr;
    entries[2].textureView = shadowMapView;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].buffer = nullptr;
    entries[3].offset = 0;
    entries[3].size = 0;
    entries[3].sampler = envSampler;
    entries[3].textureView = nullptr;

    entries[4].nextInChain = nullptr;
    entries[4].binding = 4;
    entries[4].buffer = nullptr;
    entries[4].offset = 0;
    entries[4].size = 0;
    entries[4].sampler = nullptr;
    entries[4].textureView = envMapView;

    entries[5].nextInChain = nullptr;
    entries[5].binding = 5;
    entries[5].buffer = pointLightsBuffer;
    entries[5].offset = 0;
    entries[5].size = wgpuBufferGetSize(pointLightsBuffer);
    entries[5].sampler = nullptr;
    entries[5].textureView = nullptr;

    entries[6].nextInChain = nullptr;
    entries[6].binding = 6;
    entries[6].buffer = nullptr;
    entries[6].offset = 0;
    entries[6].size = 0;
    entries[6].sampler = nullptr;
    entries[6].textureView = noise3DView;

    entries[7].nextInChain = nullptr;
    entries[7].binding = 7;
    entries[7].buffer = nullptr;
    entries[7].offset = 0;
    entries[7].size = 0;
    entries[7].sampler = noise3DSampler;
    entries[7].textureView = nullptr;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 8;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createGenMipmapBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input, WGPUTextureView output)
{
    WGPUBindGroupEntry entries[2];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = 0;
    entries[0].offset = 0;
    entries[0].size = 0;
    entries[0].sampler = nullptr;
    entries[0].textureView = input;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = 0;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = output;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 2;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createBlurShadowBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input, WGPUTextureView output)
{
    WGPUBindGroupEntry entries[2];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = 0;
    entries[0].offset = 0;
    entries[0].size = 0;
    entries[0].sampler = nullptr;
    entries[0].textureView = input;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = 0;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = output;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 2;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createWaterBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView hdrColorTexture, WGPUTextureView hdrDepthTexture,
    WGPUTextureView waterDataTexture, WGPUBuffer uniformBuffer)
{
    WGPUBindGroupEntry entries[4];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = 0;
    entries[0].offset = 0;
    entries[0].size = 0;
    entries[0].sampler = nullptr;
    entries[0].textureView = hdrColorTexture;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = 0;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = hdrDepthTexture;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].buffer = 0;
    entries[2].offset = 0;
    entries[2].size = 0;
    entries[2].sampler = nullptr;
    entries[2].textureView = waterDataTexture;

    entries[3].nextInChain = nullptr;
    entries[3].binding = 3;
    entries[3].buffer = uniformBuffer;
    entries[3].offset = 0;
    entries[3].size = wgpuBufferGetSize(uniformBuffer);
    entries[3].sampler = nullptr;
    entries[3].textureView = nullptr;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 4;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createSimulateWaterBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer, WGPUTextureView input, WGPUTextureView output)
{
    WGPUBindGroupEntry entries[3];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = wgpuBufferGetSize(uniformBuffer);
    entries[0].sampler = nullptr;
    entries[0].textureView = nullptr;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = nullptr;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = input;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].buffer = nullptr;
    entries[2].offset = 0;
    entries[2].size = 0;
    entries[2].sampler = nullptr;
    entries[2].textureView = output;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 3;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createHDRBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUTextureView input)
{
    WGPUBindGroupEntry entries[1];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = 0;
    entries[0].offset = 0;
    entries[0].size = 0;
    entries[0].sampler = nullptr;
    entries[0].textureView = input;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 1;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUBindGroup createTextBindGroup(WGPUDevice device, WGPUBindGroupLayout bindGroupLayout, WGPUBuffer uniformBuffer, WGPUTextureView fontTexture, WGPUSampler fontSampler)
{
    WGPUBindGroupEntry entries[3];

    entries[0].nextInChain = nullptr;
    entries[0].binding = 0;
    entries[0].buffer = uniformBuffer;
    entries[0].offset = 0;
    entries[0].size = wgpuBufferGetSize(uniformBuffer);
    entries[0].sampler = nullptr;
    entries[0].textureView = nullptr;

    entries[1].nextInChain = nullptr;
    entries[1].binding = 1;
    entries[1].buffer = nullptr;
    entries[1].offset = 0;
    entries[1].size = 0;
    entries[1].sampler = nullptr;
    entries[1].textureView = fontTexture;

    entries[2].nextInChain = nullptr;
    entries[2].binding = 2;
    entries[2].buffer = nullptr;
    entries[2].offset = 0;
    entries[2].size = 0;
    entries[2].sampler = fontSampler;
    entries[2].textureView = nullptr;

    WGPUBindGroupDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.layout = bindGroupLayout;
    descriptor.entryCount = 3;
    descriptor.entries = entries;

    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

WGPUTextureView createTextureView(WGPUTexture texture, int level)
{
    return createTextureView(texture, level, wgpuTextureGetFormat(texture));
}

WGPUTextureView createTextureView(WGPUTexture texture, int level, WGPUTextureFormat format)
{
    WGPUTextureViewDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.format = format;
    descriptor.dimension = WGPUTextureViewDimension_2D;
    descriptor.baseMipLevel = level;
    descriptor.mipLevelCount = 1;
    descriptor.baseArrayLayer = 0;
    descriptor.arrayLayerCount = 1;
    descriptor.aspect = WGPUTextureAspect_All;

    return wgpuTextureCreateView(texture, &descriptor);
}

WGPUCommandEncoder createCommandEncoder(WGPUDevice device)
{
    WGPUCommandEncoderDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;

    return wgpuDeviceCreateCommandEncoder(device, &descriptor);
}

WGPURenderPassEncoder createMainRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget, WGPUTextureView resolveTarget, glm::vec4 const & clearColor)
{
    WGPURenderPassColorAttachment colorAttachment;
    colorAttachment.nextInChain = nullptr;
    colorAttachment.view = colorTarget;
    colorAttachment.resolveTarget = resolveTarget;
    colorAttachment.loadOp = WGPULoadOp_Load;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {clearColor.r, clearColor.g, clearColor.b, clearColor.a};

    WGPURenderPassDepthStencilAttachment depthStencilAttachment;
    depthStencilAttachment.view = depthTarget;
    depthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
    depthStencilAttachment.depthClearValue = 1.f;
    depthStencilAttachment.depthReadOnly = false;
    depthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
    depthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
    depthStencilAttachment.stencilClearValue = 0;
    depthStencilAttachment.stencilReadOnly = true;

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &colorAttachment;
    descriptor.depthStencilAttachment = &depthStencilAttachment;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPURenderPassEncoder createWaterRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, glm::vec4 const & clearColor)
{
    WGPURenderPassColorAttachment colorAttachment;
    colorAttachment.nextInChain = nullptr;
    colorAttachment.view = colorTarget;
    colorAttachment.resolveTarget = nullptr;
    colorAttachment.loadOp = WGPULoadOp_Load;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {clearColor.r, clearColor.g, clearColor.b, clearColor.a};

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &colorAttachment;
    descriptor.depthStencilAttachment = nullptr;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPURenderPassEncoder createShadowRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView depthTarget)
{
    WGPURenderPassDepthStencilAttachment depthStencilAttachment;
    depthStencilAttachment.view = depthTarget;
    depthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
    depthStencilAttachment.depthClearValue = 1.f;
    depthStencilAttachment.depthReadOnly = false;
    depthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
    depthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
    depthStencilAttachment.stencilClearValue = 0;
    depthStencilAttachment.stencilReadOnly = true;

    WGPURenderPassColorAttachment colorAttachments[1];
    colorAttachments[0].nextInChain = nullptr;
    colorAttachments[0].view = colorTarget;
    colorAttachments[0].resolveTarget = nullptr;
    colorAttachments[0].loadOp = WGPULoadOp_Clear;
    colorAttachments[0].storeOp = WGPUStoreOp_Store;
    colorAttachments[0].clearValue = {1.0, 1.0, 1.0, 1.0};

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = colorAttachments;
    descriptor.depthStencilAttachment = &depthStencilAttachment;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPURenderPassEncoder createEnvRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView colorTarget, WGPUTextureView resolveTarget)
{
    WGPURenderPassColorAttachment colorAttachment;
    colorAttachment.nextInChain = nullptr;
    colorAttachment.view = colorTarget;
    colorAttachment.resolveTarget = resolveTarget;
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0.0, 0.0, 0.0, 0.0};

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &colorAttachment;
    descriptor.depthStencilAttachment = nullptr;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPURenderPassEncoder createLDRRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView target)
{
    WGPURenderPassColorAttachment colorAttachment;
    colorAttachment.nextInChain = nullptr;
    colorAttachment.view = target;
    colorAttachment.resolveTarget = nullptr;
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0.0, 0.0, 0.0, 0.0};

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &colorAttachment;
    descriptor.depthStencilAttachment = nullptr;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPURenderPassEncoder createTextRenderPass(WGPUCommandEncoder commandEncoder, WGPUTextureView target)
{
    WGPURenderPassColorAttachment colorAttachment;
    colorAttachment.nextInChain = nullptr;
    colorAttachment.view = target;
    colorAttachment.resolveTarget = nullptr;
    colorAttachment.loadOp = WGPULoadOp_Load;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0.0, 0.0, 0.0, 0.0};

    WGPURenderPassDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &colorAttachment;
    descriptor.depthStencilAttachment = nullptr;
    descriptor.occlusionQuerySet = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginRenderPass(commandEncoder, &descriptor);
}

WGPUComputePassEncoder createComputePass(WGPUCommandEncoder commandEncoder)
{
    WGPUComputePassDescriptor descriptor;
    descriptor.label = nullptr;
    descriptor.nextInChain = nullptr;
    descriptor.timestampWrites = nullptr;

    return wgpuCommandEncoderBeginComputePass(commandEncoder, &descriptor);
}

WGPUCommandBuffer commandEncoderFinish(WGPUCommandEncoder commandEncoder)
{
    WGPUCommandBufferDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;

    return wgpuCommandEncoderFinish(commandEncoder, &descriptor);
}

WGPUSampler createDefaultSampler(WGPUDevice device)
{
    WGPUSamplerDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.addressModeU = WGPUAddressMode_MirrorRepeat;
    descriptor.addressModeV = WGPUAddressMode_MirrorRepeat;
    descriptor.addressModeW = WGPUAddressMode_MirrorRepeat;
    descriptor.magFilter = WGPUFilterMode_Linear;
    descriptor.minFilter = WGPUFilterMode_Linear;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
    descriptor.lodMinClamp = 0.f;
    descriptor.lodMaxClamp = 255.f;
    descriptor.compare = WGPUCompareFunction_Undefined;
    descriptor.maxAnisotropy = 16;

    return wgpuDeviceCreateSampler(device, &descriptor);
}

WGPUSampler createShadowSampler(WGPUDevice device)
{
    WGPUSamplerDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.addressModeU = WGPUAddressMode_ClampToEdge;
    descriptor.addressModeV = WGPUAddressMode_ClampToEdge;
    descriptor.addressModeW = WGPUAddressMode_ClampToEdge;
    descriptor.magFilter = WGPUFilterMode_Linear;
    descriptor.minFilter = WGPUFilterMode_Linear;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    descriptor.lodMinClamp = 0.f;
    descriptor.lodMaxClamp = 0.f;
    descriptor.compare = WGPUCompareFunction_Undefined;
    descriptor.maxAnisotropy = 1;

    return wgpuDeviceCreateSampler(device, &descriptor);
}

WGPUSampler createEnvSampler(WGPUDevice device)
{
    WGPUSamplerDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.addressModeU = WGPUAddressMode_Repeat;
    descriptor.addressModeV = WGPUAddressMode_ClampToEdge;
    descriptor.addressModeW = WGPUAddressMode_ClampToEdge;
    descriptor.magFilter = WGPUFilterMode_Linear;
    descriptor.minFilter = WGPUFilterMode_Linear;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
    descriptor.lodMinClamp = 0.f;
    descriptor.lodMaxClamp = 255.f;
    descriptor.compare = WGPUCompareFunction_Undefined;
    descriptor.maxAnisotropy = 1;

    return wgpuDeviceCreateSampler(device, &descriptor);
}

WGPUSampler create3DNoiseSampler(WGPUDevice device)
{
    WGPUSamplerDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.addressModeU = WGPUAddressMode_Repeat;
    descriptor.addressModeV = WGPUAddressMode_Repeat;
    descriptor.addressModeW = WGPUAddressMode_Repeat;
    descriptor.magFilter = WGPUFilterMode_Linear;
    descriptor.minFilter = WGPUFilterMode_Linear;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    descriptor.lodMinClamp = 0.f;
    descriptor.lodMaxClamp = 0.f;
    descriptor.compare = WGPUCompareFunction_Undefined;
    descriptor.maxAnisotropy = 1;

    return wgpuDeviceCreateSampler(device, &descriptor);
}

WGPUSampler createTextSampler(WGPUDevice device)
{
    WGPUSamplerDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.addressModeU = WGPUAddressMode_ClampToEdge;
    descriptor.addressModeV = WGPUAddressMode_ClampToEdge;
    descriptor.addressModeW = WGPUAddressMode_ClampToEdge;
    descriptor.magFilter = WGPUFilterMode_Nearest;
    descriptor.minFilter = WGPUFilterMode_Nearest;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    descriptor.lodMinClamp = 0.f;
    descriptor.lodMaxClamp = 0.f;
    descriptor.compare = WGPUCompareFunction_Undefined;
    descriptor.maxAnisotropy = 1;

    return wgpuDeviceCreateSampler(device, &descriptor);
}

WGPUTexture createWhiteTexture(WGPUDevice device, WGPUQueue queue)
{
    auto sRGBViewFormat = WGPUTextureFormat_RGBA8UnormSrgb;

    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {1, 1, 1};
    textureDescriptor.format = WGPUTextureFormat_RGBA8Unorm;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 1;
    textureDescriptor.viewFormats = &sRGBViewFormat;

    auto texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

    std::vector<unsigned char> pixels(4, 255);

    WGPUImageCopyTexture imageCopyTexture;
    imageCopyTexture.nextInChain = nullptr;
    imageCopyTexture.texture = texture;
    imageCopyTexture.mipLevel = 0;
    imageCopyTexture.origin = {0, 0, 0};
    imageCopyTexture.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout textureDataLayout;
    textureDataLayout.nextInChain = nullptr;
    textureDataLayout.offset = 0;
    textureDataLayout.bytesPerRow = 4;
    textureDataLayout.rowsPerImage = 1;

    WGPUExtent3D writeSize;
    writeSize.width = 1;
    writeSize.height = 1;
    writeSize.depthOrArrayLayers = 1;

    wgpuQueueWriteTexture(queue, &imageCopyTexture, pixels.data(), pixels.size(), &textureDataLayout, &writeSize);

    return texture;
}

WGPUTexture createShadowMapTexture(WGPUDevice device, std::uint32_t size)
{
    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_StorageBinding;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {size, size, 1};
    textureDescriptor.format = WGPUTextureFormat_R32Float;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 0;
    textureDescriptor.viewFormats = nullptr;

    return wgpuDeviceCreateTexture(device, &textureDescriptor);
}

WGPUTexture createShadowMapDepthTexture(WGPUDevice device, std::uint32_t size)
{
    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_RenderAttachment;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {size, size, 1};
    textureDescriptor.format = WGPUTextureFormat_Depth24Plus;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 0;
    textureDescriptor.viewFormats = nullptr;

    return wgpuDeviceCreateTexture(device, &textureDescriptor);
}

WGPUTexture createStubEnvTexture(WGPUDevice device, WGPUQueue queue)
{
    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {1, 2, 1};
    textureDescriptor.format = WGPUTextureFormat_RGBA32Float;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 0;
    textureDescriptor.viewFormats = nullptr;

    auto texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

    std::vector<float> pixels =
    {
        0.4f, 0.7f, 1.f, 1.f,
        0.f, 0.f, 0.f, 1.f,
    };

    WGPUImageCopyTexture imageCopyTexture;
    imageCopyTexture.nextInChain = nullptr;
    imageCopyTexture.texture = texture;
    imageCopyTexture.mipLevel = 0;
    imageCopyTexture.origin = {0, 0, 0};
    imageCopyTexture.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout textureDataLayout;
    textureDataLayout.nextInChain = nullptr;
    textureDataLayout.offset = 0;
    textureDataLayout.bytesPerRow = 16;
    textureDataLayout.rowsPerImage = 2;

    WGPUExtent3D writeSize;
    writeSize.width = 1;
    writeSize.height = 2;
    writeSize.depthOrArrayLayers = 1;

    wgpuQueueWriteTexture(queue, &imageCopyTexture, pixels.data(), pixels.size() * sizeof(pixels[0]), &textureDataLayout, &writeSize);

    return texture;
}

WGPUTexture create3DNoiseTexture(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & path)
{
    std::vector<std::uint8_t> noiseData(std::filesystem::file_size(path));
    {
        std::ifstream input(path, std::ios::binary);
        input.read(reinterpret_cast<char *>(noiseData.data()), noiseData.size());
    }

    // Assume cubic texture
    std::uint32_t textureSize = std::pow<float>(noiseData.size(), 1.f / 3.f);
    if (textureSize * textureSize * textureSize != noiseData.size())
        throw std::runtime_error("3D noise texture must have all dimensions equal");

    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
    textureDescriptor.dimension = WGPUTextureDimension_3D;
    textureDescriptor.size = {textureSize, textureSize, textureSize};
    textureDescriptor.format = WGPUTextureFormat_R8Unorm;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 0;
    textureDescriptor.viewFormats = nullptr;

    auto texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

    WGPUImageCopyTexture imageCopyTexture;
    imageCopyTexture.nextInChain = nullptr;
    imageCopyTexture.texture = texture;
    imageCopyTexture.mipLevel = 0;
    imageCopyTexture.origin = {0, 0, 0};
    imageCopyTexture.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout textureDataLayout;
    textureDataLayout.nextInChain = nullptr;
    textureDataLayout.offset = 0;
    textureDataLayout.bytesPerRow = textureSize;
    textureDataLayout.rowsPerImage = textureSize;

    WGPUExtent3D writeSize;
    writeSize.width = textureSize;
    writeSize.height = textureSize;
    writeSize.depthOrArrayLayers = textureSize;

    wgpuQueueWriteTexture(queue, &imageCopyTexture, noiseData.data(), noiseData.size() * sizeof(noiseData[0]), &textureDataLayout, &writeSize);

    return texture;
}

WGPUTextureView create3DNoiseTextureView(WGPUTexture texture)
{
    WGPUTextureViewDescriptor descriptor;
    descriptor.nextInChain = nullptr;
    descriptor.label = nullptr;
    descriptor.format = WGPUTextureFormat_R8Unorm;
    descriptor.dimension = WGPUTextureViewDimension_3D;
    descriptor.baseMipLevel = 0;
    descriptor.mipLevelCount = 1;
    descriptor.baseArrayLayer = 0;
    descriptor.arrayLayerCount = 1;
    descriptor.aspect = WGPUTextureAspect_All;

    return wgpuTextureCreateView(texture, &descriptor);
}

WGPUTexture createFontTexture(WGPUDevice device, WGPUQueue queue, std::filesystem::path const & path)
{
    auto imageData = glTF::loadImage(path);

    WGPUTextureDescriptor textureDescriptor;
    textureDescriptor.nextInChain = nullptr;
    textureDescriptor.label = nullptr;
    textureDescriptor.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {(std::uint32_t)imageData.width, (std::uint32_t)imageData.height, 1};
    textureDescriptor.format = WGPUTextureFormat_R8Unorm;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.viewFormatCount = 0;
    textureDescriptor.viewFormats = nullptr;

    auto texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

    WGPUImageCopyTexture imageCopyTexture;
    imageCopyTexture.nextInChain = nullptr;
    imageCopyTexture.texture = texture;
    imageCopyTexture.mipLevel = 0;
    imageCopyTexture.origin = {0, 0, 0};
    imageCopyTexture.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout textureDataLayout;
    textureDataLayout.nextInChain = nullptr;
    textureDataLayout.offset = 0;
    textureDataLayout.bytesPerRow = imageData.width;
    textureDataLayout.rowsPerImage = imageData.height;

    WGPUExtent3D writeSize;
    writeSize.width = imageData.width;
    writeSize.height = imageData.height;
    writeSize.depthOrArrayLayers = 1;

    wgpuQueueWriteTexture(queue, &imageCopyTexture, imageData.data.get(), imageData.width * imageData.height, &textureDataLayout, &writeSize);

    return texture;
}
