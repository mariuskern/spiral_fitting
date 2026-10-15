#pragma once

#include <map>
#include <vector>
#include <set>

void OmissionTest(AlignmentMap *am, std::map<int,Patch> *patches, std::vector<int> patchNums,
            std::set<int> badPatches, std::set<std::pair<int,int>> manualBadRel,
            std::set<int> badBridges, int N);
