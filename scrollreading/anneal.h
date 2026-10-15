// Modified in October 2026 to additionally accept a const PatchConfig reference.

#pragma once

#include "patch_config.h"

// void Anneal(AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums, std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel, std::set<int> badBridges, int iterations, int N, float initT0=-1.0);
void Anneal(const PatchConfig& config, AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums, std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel, std::set<int> badBridges, int iterations, int N, float initT0=-1.0);

// void AnnealAll(int numComponents,AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums,
//             std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel,
//             std::set<int> badBridges, int iterations, int N, float initT0=-1.0);
void AnnealAll(const PatchConfig& config, int numComponents,AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums,
            std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel,
            std::set<int> badBridges, int iterations, int N, float initT0=-1.0);

