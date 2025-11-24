// PoseVisualizationWidget.h
// UMG Widget for displaying YOLO pose estimation (17 keypoints + bbox)

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "PoseVisualizationWidget.generated.h"

UCLASS()
class AIRUNTIME_API UPoseVisualizationWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    UPoseVisualizationWidget(const FObjectInitializer& ObjectInitializer);

    // Update with new pose data
    // JointPositions layout for YOLO:
    // [0] = Box center (X, Y, 0)
    // [1] = Box size (Width, Height, 0)
    // [2-18] = 17 keypoints (X, Y, Visibility)
    UFUNCTION(BlueprintCallable, Category = "Pose Visualization")
    void UpdatePoseData(const TArray<FVector>& JointPositions, float Confidence);

    // Alias for compatibility
    UFUNCTION(BlueprintCallable, Category = "Pose Visualization")
    void UpdateDetection(const TArray<FVector>& JointPositions, float Confidence);

    // Clear current detection
    UFUNCTION(BlueprintCallable, Category = "Pose Visualization")
    void ClearDetection();

    // Visualization options
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    bool bShowBoundingBox;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    bool bShowKeypoints;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    bool bShowSkeleton;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    bool bShowConfidence;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    float KeypointSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    float BoundingBoxThickness;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    float SkeletonThickness;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    FLinearColor BoundingBoxColor;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    FLinearColor KeypointColor;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Style")
    FLinearColor SkeletonColor;

    // Coordinate transformation
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Transform")
    bool bFlipHorizontal;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Transform")
    bool bFlipVertical;

    // Animation
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Animation")
    bool bEnableSmoothing;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose Visualization|Animation")
    float SmoothingFactor;

protected:
    virtual void NativeConstruct() override;
    virtual int32 NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
        const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
        int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;

private:
    // Current detection state
    bool bHasValidDetection;
    float CurrentConfidence;

    // Bounding box (normalized coordinates)
    FVector2D CurrentBoxCenter;
    FVector2D CurrentBoxSize;

    // 17 keypoints (normalized coordinates)
    TArray<FVector2D> CurrentKeypoints;
    TArray<float> CurrentKeypointVisibility;
};