// Modified in October 2026 to additionally accept a const PatchConfig reference.

#pragma once

#include "patch_config.h"

#include <map>
#include <vector>
#include <set>

// void OmissionTest(AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums,
//             std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel,
//             std::set<int> badBridges, int N);
void OmissionTest(const PatchConfig& config, AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums,
            std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel,
            std::set<int> badBridges, int N);
