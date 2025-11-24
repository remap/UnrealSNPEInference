// AIInferenceActor.h
// AI Inference Actor - Header file
// UPDATED: For YOLO11n-pose model
// Input: 256x256x3
// Output: output_0 (1x56x1344) - 17 keypoints + bbox

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AIInferenceActor.generated.h"

// Struct to hold inference results
USTRUCT(BlueprintType)
struct FAIInferenceResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AI Inference")
    bool bSuccess;

    // JointPositions layout for YOLO pose:
    // [0] = Box center (X, Y, 0)
    // [1] = Box size (Width, Height, 0)
    // [2-18] = 17 keypoints (X, Y, Visibility)
    //   [2] = Nose
    //   [3] = Left Eye
    //   [4] = Right Eye
    //   [5] = Left Ear
    //   [6] = Right Ear
    //   [7] = Left Shoulder
    //   [8] = Right Shoulder
    //   [9] = Left Elbow
    //   [10] = Right Elbow
    //   [11] = Left Wrist
    //   [12] = Right Wrist
    //   [13] = Left Hip
    //   [14] = Right Hip
    //   [15] = Left Knee
    //   [16] = Right Knee
    //   [17] = Left Ankle
    //   [18] = Right Ankle
    UPROPERTY(BlueprintReadOnly, Category = "AI Inference")
    TArray<FVector> JointPositions;

    UPROPERTY(BlueprintReadOnly, Category = "AI Inference")
    float Confidence;

    UPROPERTY(BlueprintReadOnly, Category = "AI Inference")
    int32 ProcessingTimeMS;

    FAIInferenceResult()
        : bSuccess(false)
        , Confidence(0.0f)
        , ProcessingTimeMS(0)
    {
    }
};

UCLASS()
class AIRUNTIME_API AAIInferenceActor : public AActor
{
    GENERATED_BODY()

public:
    AAIInferenceActor();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;

    // Initialize the inference system with a model
    UFUNCTION(BlueprintCallable, Category = "AI Inference")
    bool InitializeInference(const FString& ModelName);

    // Process a camera frame and return pose estimation
    UFUNCTION(BlueprintCallable, Category = "AI Inference")
    FAIInferenceResult ProcessCameraFrame(const TArray<uint8>& RGBData, int32 Width, int32 Height);

    // Shutdown the inference system
    UFUNCTION(BlueprintCallable, Category = "AI Inference")
    void ShutdownInference();

    // Configuration properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Inference")
    bool bAutoInitialize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Inference")
    FString DefaultModelName;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Inference")
    FString ModelDirectory;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Inference")
    bool bEnableLogging;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Inference")
    bool bUseGPUAcceleration;

    bool SaveFrames;

    // Blueprint events
    /** Called when inference completes successfully */
    UFUNCTION(BlueprintImplementableEvent, Category = "AI Inference")
    void OnInferenceCompleted(const FAIInferenceResult& Result);

    /** Called when inference fails */
    UFUNCTION(BlueprintImplementableEvent, Category = "AI Inference")
    void OnInferenceFailed(const FString& ErrorMessage);

private:
    // Internal state
    bool bIsInitialized;
    FString LoadedModelPath;
    int32 InferenceCounter;
    double TotalInferenceTime;

    // QAIRT/SNPE pointers (opaque to avoid header pollution)
    void* WorkspacePtr;
    void* GraphRunnerPtr;

    // Helper functions
    bool EnsureModelInstalled(const FString& ModelName);
    bool RunInference(const TArray<float>& InputData, TArray<float>& OutputData);
    TArray<float> PreprocessImageData(const TArray<uint8>& RGBData, int32 Width, int32 Height);
    FAIInferenceResult PostprocessOutput(const TArray<float>& OutputData, int32 CameraWidth, int32 CameraHeight);
    FAIInferenceResult PostprocessOutputRaw(const TArray<float>& OutputData, int32 CameraWidth, int32 CameraHeight);

    // Debug functions
    void SaveDebugImage(const TArray<float>& ImageData, int32 Width, int32 Height, const FString& Filename);
    void SaveKeypointsDebugImage(const FAIInferenceResult& Result, int32 Width, int32 Height, const FString& Filename);

};
