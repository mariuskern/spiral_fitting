#pragma once

#include <filesystem>
#include <string>

#include "parameters.h"

struct PatchConfig {
    int seedX = SEED_X;
    int seedY = SEED_Y;
    int seedZ = SEED_Z;

    double axis1X = SEED_AXIS1_X;
    double axis1Y = SEED_AXIS1_Y;
    double axis1Z = SEED_AXIS1_Z;

    double axis2X = SEED_AXIS2_X;
    double axis2Y = SEED_AXIS2_Y;
    double axis2Z = SEED_AXIS2_Z;

    std::string outputDir = OUTPUT_DIR;
};


// static std::string OutputPath(const PatchConfig& config, const std::string& filename)
inline std::string OutputPath(const PatchConfig& config, const std::string& filename)
{
    return (std::filesystem::path(config.outputDir) / filename).string();
}
