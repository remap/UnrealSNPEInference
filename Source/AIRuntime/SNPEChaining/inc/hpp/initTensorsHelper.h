#if PLATFORM_ANDROID 
//
// Created by Chiheb Boussema on 24/9/25.
//

#ifndef SNPECHAININGDEMO_INITTENSORSHELPER_H
#define SNPECHAININGDEMO_INITTENSORSHELPER_H

#include <string>
#include <vector>
#include <unistd.h>
#include <unordered_map>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlEnums.hpp"

#include "inc/hpp/TensorWorkspace.hpp"
#include "inc/hpp/ModelSession.hpp"
#include "inc/hpp/GraphRunner.hpp"
#include "inc/hpp/ParseConfig.hpp"
#include "inc/hpp/MMapFile.h"

#include <unordered_set>   // std::unordered_set
#include <random>          // std::mt19937, std::normal_distribution
#include <fstream>         // std::ifstream  (fixes “undefined template basic_ifstream”)
#include <cstring>         // std::memset / std::memcmp
#include <cctype>          // std::isspace
#include <chrono>          // seeding from steady_clock

static std::unordered_set<std::string> collectProducedNames(const PipelineCfg& cfg);

static std::unordered_set<std::string> collectConsumedNames(const PipelineCfg& cfg);

static std::vector<std::string> computeGraphRoots(const PipelineCfg& cfg);

static void fillConst(void* p, size_t bytes, float value);

static void fillRandom(void* p, size_t bytes, float mean, float stddev, uint32_t seed);

static bool readFileToBuffer(const std::string& path, void* dst, size_t bytes);

static bool readAssetToBuffer(AAssetManager* mgr, const char* asset, void* dst, size_t bytes);

static bool seedOneTensor(TensorWorkspace& ws,
                          const std::string& wsName,
                          const InitSpec* spec,
                          AAssetManager* mgr,
                          std::string* emsg);

bool seedRequiredInputs(const PipelineCfg& cfg,
                               TensorWorkspace& ws,
                               AAssetManager* mgr,
                               std::string* emsg);

#endif //SNPECHAININGDEMO_INITTENSORSHELPER_H
#endif