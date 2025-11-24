// AIInferenceActor.cpp
// AI Inference Actor - Receives camera frames and runs pose estimation inference
// UPDATED: For YOLO11n-pose model
// Input: 256x256x3
// Output: output_0 (1x56x1344) - detections with bbox + keypoints

#include "AIInferenceActor.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"

#if PLATFORM_ANDROID
#include "Android/AndroidApplication.h"
#include "Android/AndroidJNI.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// QAIRT/SNPE includes
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "inc/hpp/inference.h"
#include "inc/hpp/ModelSession.hpp"
#include "inc/hpp/GraphRunner.hpp"
#include "inc/hpp/TensorWorkspace.hpp"
#include "inc/hpp/MMapAsset.hpp"
#include "inc/hpp/ParseConfig.hpp"
#include "inc/hpp/MMapFile.h"
#include "inc/hpp/newInferenceHelper.hpp"

#define LOG_TAG_AI "AI_INFERENCE"
#define LOGE_AI(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_AI, __VA_ARGS__)
#define LOGI_AI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG_AI, __VA_ARGS__)
#define LOGW_AI(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG_AI, __VA_ARGS__)
#else
// Fallback macros for non-Android platforms (Windows/Mac editor)
#define LOGE_AI(...) 
#define LOGI_AI(...) 
#define LOGW_AI(...) 
#endif

AAIInferenceActor::AAIInferenceActor()
{
    PrimaryActorTick.bCanEverTick = true;

    // Default configuration
    bAutoInitialize = true;
    DefaultModelName = TEXT("yolo11n-pose.dlc");
    bEnableLogging = true;
    bUseGPUAcceleration = false;

    // Internal state
    bIsInitialized = false;
    InferenceCounter = 0;
    TotalInferenceTime = 0.0;

#if PLATFORM_ANDROID
    WorkspacePtr = nullptr;
    GraphRunnerPtr = nullptr;
#endif
}

void AAIInferenceActor::BeginPlay()
{
    Super::BeginPlay();

    if (bAutoInitialize && !DefaultModelName.IsEmpty())
    {
        UE_LOG(LogTemp, Log, TEXT("Auto-initializing AI inference with model: %s"), *DefaultModelName);
        InitializeInference(DefaultModelName);
    }
}

void AAIInferenceActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Optional: Add periodic status logging
    if (bEnableLogging && InferenceCounter > 0 && InferenceCounter % 100 == 0)
    {
        float AvgTime = TotalInferenceTime / InferenceCounter;
        UE_LOG(LogTemp, Log, TEXT("AI Inference Stats: %d inferences, avg %.2f ms"),
            InferenceCounter, AvgTime);
    }
}

bool AAIInferenceActor::InitializeInference(const FString& ModelName)
{
    if (bIsInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("AI Inference already initialized"));
        return true;
    }

    UE_LOG(LogTemp, Log, TEXT("Initializing AI Inference with model: %s"), *ModelName);

    // Ensure model is installed
    if (!EnsureModelInstalled(ModelName))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to install model: %s"), *ModelName);
        OnInferenceFailed(FString::Printf(TEXT("Model installation failed: %s"), *ModelName));
        return false;
    }

    // Get model path
    const FString ModelsDir = FPaths::ProjectPersistentDownloadDir() / TEXT("Models");
    LoadedModelPath = ModelsDir / ModelName;

#if PLATFORM_ANDROID
    LOGI_AI("Initializing QAIRT/SNPE runtime...");

    // Set ModelDirectory to where the model is installed
    ModelDirectory = ModelsDir;

    // Log SNPE version
    auto Version = SNPE::SNPEFactory::getLibraryVersion();
    UE_LOG(LogTemp, Log, TEXT("SNPE Library Version: %s"), UTF8_TO_TCHAR(Version.asString().c_str()));
    LOGI_AI("SNPE Library Version: %s", Version.asString().c_str());

    // Get JNI environment
    JNIEnv* Env = FAndroidApplication::GetJavaEnv();
    jobject Activity = FAndroidApplication::GetGameActivityThis();
    jclass ActivityClass = Env->GetObjectClass(Activity);
    jmethodID GetAssets = Env->GetMethodID(ActivityClass, "getAssets", "()Landroid/content/res/AssetManager;");
    jobject AssetMgr = Env->CallObjectMethod(Activity, GetAssets);
    AAssetManager* AMgr = AAssetManager_fromJava(Env, AssetMgr);

    // Initialize workspace and graph runner
    WorkspacePtr = new TensorWorkspace();
    GraphRunnerPtr = new GraphRunner(*static_cast<TensorWorkspace*>(WorkspacePtr));

    TensorWorkspace* WS = static_cast<TensorWorkspace*>(WorkspacePtr);
    GraphRunner* GR = static_cast<GraphRunner*>(GraphRunnerPtr);

    // Build inference chain
    std::string ConfigFilename = "model-config.json";
    char RuntimePref = bUseGPUAcceleration ? 'G' : 'C'; // 'C' = CPU, 'G' = GPU, 'D' = DSP
    bool ResetSessions = false;

    // Convert FString to std::string for ModelDirectory
    std::string ModelDirStdString = TCHAR_TO_UTF8(*ModelDirectory);

    // YOLO11n-pose expects 256x256 input
    const char* wsName = "images";
    const int32 ModelInputSize = 256;
    const int32 bytes_needed = ModelInputSize * ModelInputSize * 3 * sizeof(float);
    WS->allocate(wsName, bytes_needed);

    UE_LOG(LogTemp, Log, TEXT("Allocated %d bytes for %dx%d input tensor"), bytes_needed, ModelInputSize, ModelInputSize);
    LOGI_AI("Allocated %d bytes for %dx%d input tensor", bytes_needed, ModelInputSize, ModelInputSize);

    std::string BuildLog = buildArbitraryChain(AMgr, ModelDirStdString, ConfigFilename, *WS, *GR, RuntimePref, ResetSessions);

    UE_LOG(LogTemp, Log, TEXT("QAIRT Build Log: %s"), UTF8_TO_TCHAR(BuildLog.c_str()));
    LOGI_AI("QAIRT Build Result: %s", BuildLog.c_str());

    // Check if initialization succeeded
    if (GR == nullptr || BuildLog.find("error") != std::string::npos || BuildLog.find("failed") != std::string::npos)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to initialize QAIRT inference chain"));
        LOGE_AI("Failed to initialize QAIRT inference chain");

        // Cleanup
        Env->DeleteLocalRef(AssetMgr);
        Env->DeleteLocalRef(ActivityClass);

        OnInferenceFailed(TEXT("QAIRT initialization failed"));
        return false;
    }

    Env->DeleteLocalRef(AssetMgr);
    Env->DeleteLocalRef(ActivityClass);

    bIsInitialized = true;
    UE_LOG(LogTemp, Log, TEXT("AI Inference initialized successfully!"));
    LOGI_AI("AI Inference initialized successfully!");
    return true;

#else
    UE_LOG(LogTemp, Error, TEXT("AI Inference is only supported on Android"));
    OnInferenceFailed(TEXT("Platform not supported"));
    return false;
#endif
}

FAIInferenceResult AAIInferenceActor::ProcessCameraFrame(const TArray<uint8>& RGBData, int32 Width, int32 Height)
{
    FAIInferenceResult Result;
    Result.bSuccess = false;
    Result.Confidence = 0.0f;
    Result.ProcessingTimeMS = 0;

    if (!bIsInitialized)
    {
        UE_LOG(LogTemp, Error, TEXT("Inference not initialized! Call InitializeInference first."));
        OnInferenceFailed(TEXT("Inference not initialized"));
        return Result;
    }

    if (RGBData.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Empty RGB data provided"));
        OnInferenceFailed(TEXT("Empty input data"));
        return Result;
    }

    double StartTime = FPlatformTime::Seconds();

    if (bEnableLogging)
    {
        UE_LOG(LogTemp, Log, TEXT("Processing frame: %dx%d (%d bytes)"), Width, Height, RGBData.Num());
    }

#if PLATFORM_ANDROID
    LOGI_AI("Processing camera frame: %dx%d", Width, Height);

    // Step 1: Preprocess image data
    TArray<float> InputData = PreprocessImageData(RGBData, Width, Height);
    if (InputData.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Preprocessing failed"));
        OnInferenceFailed(TEXT("Preprocessing failed"));
        return Result;
    }

    // Step 2: Run inference
    TArray<float> OutputData;
    if (!RunInference(InputData, OutputData))
    {
        UE_LOG(LogTemp, Error, TEXT("Inference execution failed"));
        OnInferenceFailed(TEXT("Inference execution failed"));
        return Result;
    }

    // Step 3: Postprocess output - find best detection
    Result = PostprocessOutput(OutputData, Width, Height);

    // Debug: Save synchronized preprocess and keypoints images every N frames
    // Save RAW keypoints (before aspect ratio corrections) for debugging
    static int32 DebugFrameCounter = 0;
    const int32 SaveEveryNFrames = 100;  // Save every 100 frames
    if (DebugFrameCounter % SaveEveryNFrames == 0)
    {
        int32 SaveIndex = DebugFrameCounter / SaveEveryNFrames;
        SaveDebugImage(InputData, 256, 256,
            FString::Printf(TEXT("preprocess_%d.ppm"), SaveIndex));

        // Create a raw result without aspect ratio corrections for debug visualization
        FAIInferenceResult RawResult = PostprocessOutputRaw(OutputData, Width, Height);
        SaveKeypointsDebugImage(RawResult, 256, 256,
            FString::Printf(TEXT("keypoints_%d.ppm"), SaveIndex));
    }
    DebugFrameCounter++;

    // Update performance metrics
    double ProcessingTime = (FPlatformTime::Seconds() - StartTime) * 1000.0; // Convert to ms
    Result.ProcessingTimeMS = static_cast<int32>(ProcessingTime);

    InferenceCounter++;
    TotalInferenceTime += ProcessingTime;

    if (bEnableLogging && Result.bSuccess)
    {
        UE_LOG(LogTemp, Log, TEXT("Inference completed: %.2f ms, Confidence: %.2f"),
            ProcessingTime, Result.Confidence);
        LOGI_AI("Inference completed: %.2f ms, Confidence: %.2f, %d keypoints",
            ProcessingTime, Result.Confidence, Result.JointPositions.Num());
    }

    // Notify Blueprint
    if (Result.bSuccess)
    {
        OnInferenceCompleted(Result);
    }

#else
    OnInferenceFailed(TEXT("Platform not supported"));
#endif

    return Result;
}

bool AAIInferenceActor::EnsureModelInstalled(const FString& ModelName)
{
    const FString ModelsDir = FPaths::ProjectPersistentDownloadDir() / TEXT("Models");
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    if (!PlatformFile.DirectoryExists(*ModelsDir))
    {
        PlatformFile.CreateDirectory(*ModelsDir);
        UE_LOG(LogTemp, Log, TEXT("Created models directory: %s"), *ModelsDir);
    }

    const FString DlcPath = ModelsDir / ModelName;

    if (FPaths::FileExists(DlcPath))
    {
        UE_LOG(LogTemp, Log, TEXT("Model already installed: %s"), *DlcPath);
        return true;
    }

#if PLATFORM_ANDROID
    UE_LOG(LogTemp, Log, TEXT("Copying model from APK assets: %s"), *ModelName);
    LOGI_AI("Copying model from APK assets: %s", TCHAR_TO_UTF8(*ModelName));

    JNIEnv* Env = FAndroidApplication::GetJavaEnv();
    jobject Activity = FAndroidApplication::GetGameActivityThis();
    jclass ActivityClass = Env->GetObjectClass(Activity);
    jmethodID GetAssets = Env->GetMethodID(ActivityClass, "getAssets", "()Landroid/content/res/AssetManager;");
    jobject AssetMgrObj = Env->CallObjectMethod(Activity, GetAssets);

    AAssetManager* AMgr = AAssetManager_fromJava(Env, AssetMgrObj);
    AAsset* Asset = AAssetManager_open(AMgr, TCHAR_TO_UTF8(*ModelName), AASSET_MODE_STREAMING);

    if (!Asset)
    {
        UE_LOG(LogTemp, Error, TEXT("Model not found in APK assets: %s"), *ModelName);
        LOGE_AI("Model not found in APK assets: %s", TCHAR_TO_UTF8(*ModelName));

        Env->DeleteLocalRef(AssetMgrObj);
        Env->DeleteLocalRef(ActivityClass);
        return false;
    }

    // Read asset into memory
    const off_t Length = AAsset_getLength(Asset);
    TArray<uint8> Bytes;
    Bytes.SetNumUninitialized(Length);
    AAsset_read(Asset, Bytes.GetData(), Length);
    AAsset_close(Asset);

    // Save to disk
    bool bSaved = FFileHelper::SaveArrayToFile(Bytes, *DlcPath);

    Env->DeleteLocalRef(AssetMgrObj);
    Env->DeleteLocalRef(ActivityClass);

    if (bSaved)
    {
        UE_LOG(LogTemp, Log, TEXT("Model installed successfully: %s (%ld bytes)"), *DlcPath, (long)Length);
        LOGI_AI("Model installed successfully: %ld bytes", (long)Length);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to save model to disk: %s"), *DlcPath);
        LOGE_AI("Failed to save model to disk");
    }

    return bSaved;

#else
    return false;
#endif
}

bool AAIInferenceActor::RunInference(const TArray<float>& InputData, TArray<float>& OutputData)
{
#if PLATFORM_ANDROID
    if (!WorkspacePtr || !GraphRunnerPtr)
    {
        LOGE_AI("Workspace or GraphRunner is null");
        return false;
    }

    TensorWorkspace* WS = static_cast<TensorWorkspace*>(WorkspacePtr);
    GraphRunner* GR = static_cast<GraphRunner*>(GraphRunnerPtr);

    // Copy input data to workspace
    const char* InputTensorName = "images";
    size_t BytesNeeded = InputData.Num() * sizeof(float);

    if (WS->data(InputTensorName) == nullptr)
    {
        LOGE_AI("Input tensor '%s' not found in workspace", InputTensorName);
        return false;
    }

    std::memcpy(WS->data(InputTensorName), InputData.GetData(), BytesNeeded);
    LOGI_AI("Copied %zu bytes to input tensor", BytesNeeded);

    // Execute inference
    std::string ExecutionSummary = runGraph(*GR);

    if (bEnableLogging)
    {
        UE_LOG(LogTemp, Log, TEXT("Execution Summary: %s"), UTF8_TO_TCHAR(ExecutionSummary.c_str()));
        LOGI_AI("Execution Summary: %s", ExecutionSummary.c_str());
    }

    // Extract output
    // YOLO11n-pose output: (1, 56, 1344)
    // 56 = 4 (bbox) + 1 (confidence) + 51 (17 keypoints × 3)
    // 1344 = number of detection anchors

    const char* OutputName = "output_0";

    if (WS->data(OutputName) == nullptr)
    {
        LOGE_AI("Output tensor '%s' not found in workspace", OutputName);
        return false;
    }

    // Total output size: 56 * 1344 = 75264 floats
    const size_t NumChannels = 56;
    const size_t NumAnchors = 1344;
    const size_t TotalOutputSize = NumChannels * NumAnchors;

    OutputData.SetNumUninitialized(TotalOutputSize);
    std::memcpy(OutputData.GetData(), WS->data(OutputName), TotalOutputSize * sizeof(float));

    LOGI_AI("Retrieved %zu output values (%zu channels x %zu anchors)",
        TotalOutputSize, NumChannels, NumAnchors);

    return true;

#else
    return false;
#endif
}

TArray<float> AAIInferenceActor::PreprocessImageData(const TArray<uint8>& RGBData, int32 Width, int32 Height)
{
    TArray<float> ProcessedData;

    // YOLO11n-pose expects 256x256 input in CHW format (channels first)
    // and normalized to [0, 1]
    const int32 ModelInputSize = 256;
    const int32 ExpectedSize = 3 * ModelInputSize * ModelInputSize;  // CHW: 3 x 256 x 256

    ProcessedData.SetNumUninitialized(ExpectedSize);

    // Simple resize - scale to fit model input size
    float ScaleX = static_cast<float>(Width) / ModelInputSize;
    float ScaleY = static_cast<float>(Height) / ModelInputSize;

    // Output in CHW format (Channel, Height, Width)
    // Channel 0 (R): indices 0 to 65535
    // Channel 1 (G): indices 65536 to 131071
    // Channel 2 (B): indices 131072 to 196607

    for (int32 Y = 0; Y < ModelInputSize; ++Y)
    {
        for (int32 X = 0; X < ModelInputSize; ++X)
        {
            int32 SrcX = FMath::Clamp(static_cast<int32>(X * ScaleX), 0, Width - 1);
            int32 SrcY = FMath::Clamp(static_cast<int32>(Y * ScaleY), 0, Height - 1);
            int32 SrcIndex = (SrcY * Width + SrcX) * 3;

            // CHW format: separate planes for each channel
            int32 PixelIndex = Y * ModelInputSize + X;
            int32 PlaneSize = ModelInputSize * ModelInputSize;

            // Normalize to [0, 1]
            ProcessedData[0 * PlaneSize + PixelIndex] = RGBData[SrcIndex + 0] / 255.0f; // R plane
            ProcessedData[1 * PlaneSize + PixelIndex] = RGBData[SrcIndex + 1] / 255.0f; // G plane
            ProcessedData[2 * PlaneSize + PixelIndex] = RGBData[SrcIndex + 2] / 255.0f; // B plane
        }
    }

    if (bEnableLogging)
    {
        UE_LOG(LogTemp, Log, TEXT("Preprocessed %dx%d to %dx%d (CHW format, [0-1] range)"), Width, Height, ModelInputSize, ModelInputSize);

        // Debug: Log sample values from different channels
        if (ProcessedData.Num() >= 196608)
        {
            int32 PlaneSize = ModelInputSize * ModelInputSize;
            LOGI_AI("Sample R values: %.3f %.3f %.3f",
                ProcessedData[0], ProcessedData[1], ProcessedData[2]);
            LOGI_AI("Sample G values: %.3f %.3f %.3f",
                ProcessedData[PlaneSize], ProcessedData[PlaneSize + 1], ProcessedData[PlaneSize + 2]);
            LOGI_AI("Sample B values: %.3f %.3f %.3f",
                ProcessedData[2 * PlaneSize], ProcessedData[2 * PlaneSize + 1], ProcessedData[2 * PlaneSize + 2]);
        }
    }

    return ProcessedData;
}

void AAIInferenceActor::SaveDebugImage(const TArray<float>& ImageData, int32 Width, int32 Height, const FString& Filename)
{
#if PLATFORM_ANDROID
    // Save as PPM format (simple, no dependencies)
    FString FilePath = FPaths::ProjectSavedDir() / TEXT("Debug") / Filename;

    // Ensure directory exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    PlatformFile.CreateDirectoryTree(*FPaths::GetPath(FilePath));

    // Convert CHW float [0-1] to RGB bytes
    TArray<uint8> PixelData;
    PixelData.SetNumUninitialized(Width * Height * 3);

    int32 PlaneSize = Width * Height;

    for (int32 Y = 0; Y < Height; ++Y)
    {
        for (int32 X = 0; X < Width; ++X)
        {
            int32 PixelIndex = Y * Width + X;
            int32 OutIndex = PixelIndex * 3;

            PixelData[OutIndex + 0] = static_cast<uint8>(FMath::Clamp(ImageData[0 * PlaneSize + PixelIndex] * 255.0f, 0.0f, 255.0f));
            PixelData[OutIndex + 1] = static_cast<uint8>(FMath::Clamp(ImageData[1 * PlaneSize + PixelIndex] * 255.0f, 0.0f, 255.0f));
            PixelData[OutIndex + 2] = static_cast<uint8>(FMath::Clamp(ImageData[2 * PlaneSize + PixelIndex] * 255.0f, 0.0f, 255.0f));
        }
    }

    // Write PPM file
    FString PPMHeader = FString::Printf(TEXT("P6\n%d %d\n255\n"), Width, Height);

    TArray<uint8> FileData;
    FTCHARToUTF8 HeaderConverter(*PPMHeader);
    FileData.Append((uint8*)HeaderConverter.Get(), HeaderConverter.Length());
    FileData.Append(PixelData);

    FFileHelper::SaveArrayToFile(FileData, *FilePath);

    LOGI_AI("Saved debug image: %s", TCHAR_TO_UTF8(*FilePath));
#endif
}

void AAIInferenceActor::SaveKeypointsDebugImage(const FAIInferenceResult& Result, int32 Width, int32 Height, const FString& Filename)
{
#if PLATFORM_ANDROID
    // Create a simple image with keypoints drawn
    FString FilePath = FPaths::ProjectSavedDir() / TEXT("Debug") / Filename;

    // Ensure directory exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    PlatformFile.CreateDirectoryTree(*FPaths::GetPath(FilePath));

    // Create blank image
    TArray<uint8> PixelData;
    PixelData.SetNumZeroed(Width * Height * 3);

    // Fill with dark gray background
    for (int32 i = 0; i < PixelData.Num(); i += 3)
    {
        PixelData[i] = 40;
        PixelData[i + 1] = 40;
        PixelData[i + 2] = 40;
    }

    if (Result.JointPositions.Num() >= 19)  // 2 for box + 17 keypoints
    {
        // Draw bounding box (red)
        FVector BoxCenter = Result.JointPositions[0];
        FVector BoxSize = Result.JointPositions[1];

        int32 BoxX1 = FMath::Clamp(static_cast<int32>((BoxCenter.X - BoxSize.X / 2) * Width), 0, Width - 1);
        int32 BoxY1 = FMath::Clamp(static_cast<int32>((BoxCenter.Y - BoxSize.Y / 2) * Height), 0, Height - 1);
        int32 BoxX2 = FMath::Clamp(static_cast<int32>((BoxCenter.X + BoxSize.X / 2) * Width), 0, Width - 1);
        int32 BoxY2 = FMath::Clamp(static_cast<int32>((BoxCenter.Y + BoxSize.Y / 2) * Height), 0, Height - 1);

        // Draw box edges
        for (int32 X = BoxX1; X <= BoxX2; ++X)
        {
            // Top edge
            int32 Idx = (BoxY1 * Width + X) * 3;
            PixelData[Idx] = 255; PixelData[Idx + 1] = 0; PixelData[Idx + 2] = 0;
            // Bottom edge
            Idx = (BoxY2 * Width + X) * 3;
            PixelData[Idx] = 255; PixelData[Idx + 1] = 0; PixelData[Idx + 2] = 0;
        }
        for (int32 Y = BoxY1; Y <= BoxY2; ++Y)
        {
            // Left edge
            int32 Idx = (Y * Width + BoxX1) * 3;
            PixelData[Idx] = 255; PixelData[Idx + 1] = 0; PixelData[Idx + 2] = 0;
            // Right edge
            Idx = (Y * Width + BoxX2) * 3;
            PixelData[Idx] = 255; PixelData[Idx + 1] = 0; PixelData[Idx + 2] = 0;
        }

        // Draw keypoints (green circles)
        for (int32 kp = 0; kp < 17; ++kp)
        {
            FVector Keypoint = Result.JointPositions[2 + kp];
            int32 KpX = FMath::Clamp(static_cast<int32>(Keypoint.X * Width), 0, Width - 1);
            int32 KpY = FMath::Clamp(static_cast<int32>(Keypoint.Y * Height), 0, Height - 1);

            // Draw 3x3 square for keypoint
            for (int32 dy = -1; dy <= 1; ++dy)
            {
                for (int32 dx = -1; dx <= 1; ++dx)
                {
                    int32 PX = FMath::Clamp(KpX + dx, 0, Width - 1);
                    int32 PY = FMath::Clamp(KpY + dy, 0, Height - 1);
                    int32 Idx = (PY * Width + PX) * 3;
                    PixelData[Idx] = 0; PixelData[Idx + 1] = 255; PixelData[Idx + 2] = 0;
                }
            }
        }
    }

    // Write PPM file
    FString PPMHeader = FString::Printf(TEXT("P6\n%d %d\n255\n"), Width, Height);

    TArray<uint8> FileData;
    FTCHARToUTF8 HeaderConverter(*PPMHeader);
    FileData.Append((uint8*)HeaderConverter.Get(), HeaderConverter.Length());
    FileData.Append(PixelData);

    FFileHelper::SaveArrayToFile(FileData, *FilePath);

    LOGI_AI("Saved keypoints debug image: %s", TCHAR_TO_UTF8(*FilePath));
#endif
}

FAIInferenceResult AAIInferenceActor::PostprocessOutputRaw(const TArray<float>& OutputData, int32 CameraWidth, int32 CameraHeight)
{
    // Raw version without aspect ratio corrections - for debug visualization only
    FAIInferenceResult Result;
    Result.bSuccess = false;
    Result.Confidence = 0.0f;

    const int32 NumChannels = 56;
    const int32 NumAnchors = 1344;
    const int32 NumKeypoints = 17;
    const float ModelInputSize = 256.0f;

    if (OutputData.Num() < NumChannels * NumAnchors)
    {
        return Result;
    }

    // Find best detection
    float BestScore = 0.0f;
    int32 BestIdx = -1;
    const int32 ConfidenceChannel = 55;

    for (int32 i = 0; i < NumAnchors; ++i)
    {
        float Score = OutputData[ConfidenceChannel * NumAnchors + i];
        if (Score > BestScore)
        {
            BestScore = Score;
            BestIdx = i;
        }
    }

    if (BestIdx < 0 || BestScore < 0.5f)
    {
        return Result;
    }

    // Extract raw box (no corrections)
    float BoxCenterX = OutputData[0 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxCenterY = OutputData[1 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxW = OutputData[2 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxH = OutputData[3 * NumAnchors + BestIdx] / ModelInputSize;

    Result.bSuccess = true;
    Result.Confidence = BestScore;
    Result.JointPositions.Add(FVector(BoxCenterX, BoxCenterY, 0.0f));
    Result.JointPositions.Add(FVector(BoxW, BoxH, 0.0f));

    // Extract raw keypoints (no corrections)
    for (int32 kp = 0; kp < NumKeypoints; ++kp)
    {
        int32 BaseChannel = 4 + (kp * 3);
        float KpVis = OutputData[BaseChannel * NumAnchors + BestIdx];
        float KpX = OutputData[(BaseChannel + 1) * NumAnchors + BestIdx] / ModelInputSize;
        float KpY = OutputData[(BaseChannel + 2) * NumAnchors + BestIdx] / ModelInputSize;

        KpX = FMath::Clamp(KpX, 0.0f, 1.0f);
        KpY = FMath::Clamp(KpY, 0.0f, 1.0f);
        KpVis = FMath::Clamp(KpVis, 0.0f, 1.0f);

        Result.JointPositions.Add(FVector(KpX, KpY, KpVis));
    }

    return Result;
}

FAIInferenceResult AAIInferenceActor::PostprocessOutput(const TArray<float>& OutputData, int32 CameraWidth, int32 CameraHeight)
{
    FAIInferenceResult Result;
    Result.bSuccess = false;
    Result.Confidence = 0.0f;

    // YOLO output format: (1, 56, 1344) stored as [channel][anchor]
    // Layout:
    //   Channel 0: box_center_x (all 1344 anchors)
    //   Channel 1: box_center_y
    //   Channel 2: box_width
    //   Channel 3: box_height
    //   Channels 4-54: 17 keypoints × 3 (x, y, visibility)
    //   Channel 55: confidence (NOT 4!)

    const int32 NumChannels = 56;
    const int32 NumAnchors = 1344;
    const int32 NumKeypoints = 17;
    const float ModelInputSize = 256.0f;

    if (OutputData.Num() < NumChannels * NumAnchors)
    {
        UE_LOG(LogTemp, Warning, TEXT("Output data size mismatch. Expected %d, got %d"),
            NumChannels * NumAnchors, OutputData.Num());
        return Result;
    }

    // Find best detection (highest confidence)
    // Confidence is at channel 55
    float BestScore = 0.0f;
    int32 BestIdx = -1;

    const float MinConfidenceThreshold = 0.5f;
    const int32 ConfidenceChannel = 55;

    for (int32 i = 0; i < NumAnchors; ++i)
    {
        // Index = channel * NumAnchors + anchor
        float Score = OutputData[ConfidenceChannel * NumAnchors + i];

        if (Score > BestScore)
        {
            BestScore = Score;
            BestIdx = i;
        }
    }

    if (bEnableLogging)
    {
        LOGI_AI("Best detection: anchor %d, confidence %.3f", BestIdx, BestScore);
    }

    if (BestIdx < 0 || BestScore < MinConfidenceThreshold)
    {
        if (bEnableLogging)
        {
            LOGW_AI("No valid detection found. Best score: %.3f (threshold: %.3f)", BestScore, MinConfidenceThreshold);
        }
        return Result;
    }

    // Extract best detection
    // Bbox: channels 0-3
    float BoxCenterX = OutputData[0 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxCenterY = OutputData[1 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxW = OutputData[2 * NumAnchors + BestIdx] / ModelInputSize;
    float BoxH = OutputData[3 * NumAnchors + BestIdx] / ModelInputSize;

    // The model outputs in square (1:1) space, but the display is wide (~2:1)
    // Camera was 640x480 (1.33:1), squashed to 256x256 (1:1)
    // Widget is 2246x1081 (2.08:1)
    // 
    // The Y coordinates need to be scaled to account for:
    // - Model square space to camera space: multiply by (640/480) = 1.33
    // - Camera space to widget space: multiply by (1081/2246) / (480/640) = 0.36
    // Combined: 1.33 * 0.36 = 0.48... but that would compress more
    //
    // Actually, let's think differently:
    // In model space, Y=0.5 means middle of the 256x256 image
    // That came from Y=240 in the 480-tall camera image = 240/480 = 0.5 (correct!)
    // In widget, Y=0.5 should map to 540 pixels (middle of 1081)
    // So the Y coordinates should map directly... but they're not working
    //
    // The only explanation: the camera feed on widget is NOT using the same
    // coordinate mapping. Let's try the inverse - maybe widget Y needs expansion.

    // Try: expand Y by widget aspect / camera aspect
    // Widget: 2246/1081 = 2.08, Camera: 640/480 = 1.33
    // Expansion = 2.08 / 1.33 = 1.56
    float WidgetAspect = 2246.0f / 1081.0f;
    float CameraAspect = 640.0f / 480.0f;
    float YExpansion = WidgetAspect / CameraAspect;

    // Additional correction for camera cropping/offset
    const float YOffsetCorrection = 0.0f;  // Adjust this value: try -0.05 to +0.05
    const float YScaleCorrection = 1.0f;   // Adjust this if needed: try 0.9 to 1.1

    // Apply to box height and center - expand around screen center (0.5)
    const float ScreenCenterY = 0.5f;
    BoxH = BoxH * YExpansion * YScaleCorrection;
    BoxCenterY = ScreenCenterY + (BoxCenterY - ScreenCenterY) * YExpansion * YScaleCorrection + YOffsetCorrection;

    // Clamp to valid range
    BoxCenterX = FMath::Clamp(BoxCenterX, 0.0f, 1.0f);
    BoxCenterY = FMath::Clamp(BoxCenterY, 0.0f, 1.0f);
    BoxW = FMath::Clamp(BoxW, 0.0f, 2.0f);
    BoxH = FMath::Clamp(BoxH, 0.0f, 2.0f);

    // Log the detection
    if (bEnableLogging)
    {
        LOGI_AI("=== BEST DETECTION (anchor %d, score %.3f) ===", BestIdx, BestScore);
        LOGI_AI("Box: center=(%.3f, %.3f), size=(%.3f, %.3f)", BoxCenterX, BoxCenterY, BoxW, BoxH);
    }

    // Populate result
    Result.bSuccess = true;
    Result.Confidence = BestScore;

    // Store bounding box
    Result.JointPositions.Add(FVector(BoxCenterX, BoxCenterY, 0.0f));  // Box center
    Result.JointPositions.Add(FVector(BoxW, BoxH, 0.0f));               // Box size

    // Extract 17 keypoints
    // Keypoints start at channel 4
    // YOLO pose format: each keypoint is (x, y, confidence) but stored as separate channels
    // So channels 4,5,6 = kp0_x, kp0_y, kp0_conf
    const TArray<FString> KeypointNames = {
        TEXT("Nose"), TEXT("Left Eye"), TEXT("Right Eye"), TEXT("Left Ear"), TEXT("Right Ear"),
        TEXT("Left Shoulder"), TEXT("Right Shoulder"), TEXT("Left Elbow"), TEXT("Right Elbow"),
        TEXT("Left Wrist"), TEXT("Right Wrist"), TEXT("Left Hip"), TEXT("Right Hip"),
        TEXT("Left Knee"), TEXT("Right Knee"), TEXT("Left Ankle"), TEXT("Right Ankle")
    };

    if (bEnableLogging)
    {
        LOGI_AI("Keypoints (raw channel values for anchor %d):", BestIdx);
        // Debug: print raw values for first keypoint channels
        for (int32 ch = 4; ch < 13; ++ch)
        {
            float val = OutputData[ch * NumAnchors + BestIdx];
            LOGI_AI("  Channel %d: %.3f", ch, val);
        }
    }

    for (int32 kp = 0; kp < NumKeypoints; ++kp)
    {
        int32 BaseChannel = 4 + (kp * 3);

        // Order is: visibility, X, Y (not X, Y, visibility!)
        float KpVis = OutputData[BaseChannel * NumAnchors + BestIdx];
        float KpX = OutputData[(BaseChannel + 1) * NumAnchors + BestIdx] / ModelInputSize;
        float KpY = OutputData[(BaseChannel + 2) * NumAnchors + BestIdx] / ModelInputSize;

        // Apply same Y expansion to keypoint vertical distances from SCREEN center
        // This is critical: we must expand around screen center (0.5), not box center
        // because the camera/display coordinate system is anchored at screen center
        float WidgetAspect = 2246.0f / 1081.0f;
        float CameraAspect = 640.0f / 480.0f;
        float YExpansion = WidgetAspect / CameraAspect;

        // Additional correction for camera cropping/offset
        const float YOffsetCorrection = 0.0f;  // Adjust this value: try -0.05 to +0.05
        const float YScaleCorrection = 1.0f;   // Adjust this if needed: try 0.9 to 1.1

        // Expand around screen center (0.5), not box center
        const float ScreenCenterY = 0.5f;
        KpY = ScreenCenterY + (KpY - ScreenCenterY) * YExpansion * YScaleCorrection + YOffsetCorrection;

        // Clamp coordinates
        KpX = FMath::Clamp(KpX, 0.0f, 1.0f);
        KpY = FMath::Clamp(KpY, 0.0f, 1.0f);
        KpVis = FMath::Clamp(KpVis, 0.0f, 1.0f);

        // Store keypoint with visibility in Z
        Result.JointPositions.Add(FVector(KpX, KpY, KpVis));

        if (bEnableLogging && kp < 5)  // Log first 5 keypoints
        {
            LOGI_AI("  %s: (%.3f, %.3f) vis=%.3f",
                TCHAR_TO_UTF8(*KeypointNames[kp]), KpX, KpY, KpVis);
        }
    }

    if (bEnableLogging)
    {
        LOGI_AI("Total keypoints: %d", NumKeypoints);
    }

    return Result;
}

void AAIInferenceActor::ShutdownInference()
{
    if (!bIsInitialized)
    {
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Shutting down AI Inference..."));

#if PLATFORM_ANDROID
    if (GraphRunnerPtr)
    {
        delete static_cast<GraphRunner*>(GraphRunnerPtr);
        GraphRunnerPtr = nullptr;
    }

    if (WorkspacePtr)
    {
        delete static_cast<TensorWorkspace*>(WorkspacePtr);
        WorkspacePtr = nullptr;
    }

    LOGI_AI("AI Inference shut down");
#endif

    bIsInitialized = false;

    // Log final statistics
    if (InferenceCounter > 0)
    {
        float AvgTime = TotalInferenceTime / InferenceCounter;
        UE_LOG(LogTemp, Log, TEXT("Final Stats: %d inferences, avg %.2f ms per frame"),
            InferenceCounter, AvgTime);
    }
}