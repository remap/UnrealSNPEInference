#if PLATFORM_ANDROID 
#include <jni.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "DlSystem/DlEnums.hpp"
#include "SNPE/SNPEFactory.hpp"

#include "inc/hpp/TensorWorkspace.hpp"
#include "inc/hpp/ModelSession.hpp"
#include "inc/hpp/GraphRunner.hpp"
#include "inc/hpp/ParseConfig.hpp"
#include "inc/hpp/MMapFile.h"

static DlSystem::RuntimeList makeRuntimeOrder(char pref);

static const TensorInfo* findTensor(const std::vector<TensorInfo>& v, const std::string& name);

static bool ensureWorkspaceBuffer(TensorWorkspace& ws,
                                  const std::string& wsName,
                                  size_t bytes,
                                  std::string* emsg);

static inline int64_t msSince(std::chrono::steady_clock::time_point t0);

//std::string buildModelAndGraph(AAssetManager* mgr,
//                                std::string& g_modelDir,
////                                const std::string& configJson,
//                                const PipelineCfg& cfg,
//                                const ModelCfg& mc,
//                                const char defaultRuntimePref,
//                                std::unique_ptr<TensorWorkspace>& outWs,
//                                std::unique_ptr<GraphRunner>& outGraph,
//                                std::string& log,
//                                bool reset_session=false);

std::string buildModelAndGraph(AAssetManager* mgr,
                                std::string& g_modelDir,
//                                const std::string& configJson,
                                const PipelineCfg& cfg,
                                const ModelCfg& mc,
                                const char defaultRuntimePref,
                                TensorWorkspace& outWs,
                                GraphRunner& outGraph,
                                std::string& log,
                                bool reset_session=false);

std::string buildArbitraryChain(AAssetManager* mgr,
                                std::string& g_modelDir,
                                const std::string config_filename,
                                TensorWorkspace& ws,
                                GraphRunner& gr,
                                const char defaultRuntimePref='D',
                                bool reset_sessions=false);

std::string rebuildNodeSession(GraphRunner::Node& node);
std::string rebuildMultipleNodes(std::vector<GraphRunner::Node>& nodes);
std::string rebuildAllGraphNodes(GraphRunner& gr);

std::string runGraph(GraphRunner& gr, bool reset_sessions=false);

//static bool readAssetToString(AAssetManager* mgr,
//                              const char* filename,
//                              std::string& out,
//                              std::string* emsg);
#endif