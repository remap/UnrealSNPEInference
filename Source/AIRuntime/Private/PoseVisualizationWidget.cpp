// PoseVisualizationWidget.cpp
// UMG Widget for displaying YOLO pose estimation (17 keypoints + bbox)

#include "PoseVisualizationWidget.h"
#include "Blueprint/WidgetLayoutLibrary.h"

UPoseVisualizationWidget::UPoseVisualizationWidget(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    // Default style settings
    bShowBoundingBox = true;
    bShowKeypoints = true;
    bShowSkeleton = true;
    bShowConfidence = true;

    KeypointSize = 10.0f;
    BoundingBoxThickness = 2.0f;
    SkeletonThickness = 2.0f;

    BoundingBoxColor = FLinearColor::Green;
    KeypointColor = FLinearColor::Yellow;
    SkeletonColor = FLinearColor::White;

    // Coordinate transformation
    bFlipHorizontal = false;
    bFlipVertical = false;

    // Animation
    bEnableSmoothing = true;
    SmoothingFactor = 0.3f;

    // State
    bHasValidDetection = false;
    CurrentConfidence = 0.0f;
}

void UPoseVisualizationWidget::NativeConstruct()
{
    Super::NativeConstruct();
    UE_LOG(LogTemp, Log, TEXT("PoseVisualizationWidget constructed"));
}

int32 UPoseVisualizationWidget::NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry,
    const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
    int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
    int32 MaxLayerId = Super::NativePaint(Args, AllottedGeometry, MyCullingRect, OutDrawElements,
        LayerId, InWidgetStyle, bParentEnabled);

    if (!bHasValidDetection || CurrentKeypoints.Num() < 17)
    {
        return MaxLayerId;
    }

    // Get actual widget size
    FVector2D WidgetSize = AllottedGeometry.GetLocalSize();

    // Debug: Draw coordinate grid to visualize mapping
    static bool bShowDebugGrid = true;  // Set to false to hide
    if (bShowDebugGrid)
    {
        // Draw vertical center line (X=0.5)
        TArray<FVector2D> CenterLineX;
        CenterLineX.Add(FVector2D(WidgetSize.X * 0.5f, 0));
        CenterLineX.Add(FVector2D(WidgetSize.X * 0.5f, WidgetSize.Y));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            CenterLineX, ESlateDrawEffect::None, FLinearColor::Yellow, true, 1.0f);

        // Draw horizontal center line (Y=0.5)
        TArray<FVector2D> CenterLineY;
        CenterLineY.Add(FVector2D(0, WidgetSize.Y * 0.5f));
        CenterLineY.Add(FVector2D(WidgetSize.X, WidgetSize.Y * 0.5f));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            CenterLineY, ESlateDrawEffect::None, FLinearColor::Yellow, true, 1.0f);

        // Draw quarter lines
        TArray<FVector2D> QuarterLineY1;
        QuarterLineY1.Add(FVector2D(0, WidgetSize.Y * 0.25f));
        QuarterLineY1.Add(FVector2D(WidgetSize.X, WidgetSize.Y * 0.25f));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            QuarterLineY1, ESlateDrawEffect::None, FLinearColor(1.0f, 1.0f, 0.0f, 0.3f), true, 1.0f);

        TArray<FVector2D> QuarterLineY2;
        QuarterLineY2.Add(FVector2D(0, WidgetSize.Y * 0.75f));
        QuarterLineY2.Add(FVector2D(WidgetSize.X, WidgetSize.Y * 0.75f));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            QuarterLineY2, ESlateDrawEffect::None, FLinearColor(1.0f, 1.0f, 0.0f, 0.3f), true, 1.0f);
    }

    // Draw bounding box
    if (bShowBoundingBox)
    {
        FVector2D BoxCenter = FVector2D(CurrentBoxCenter.X * WidgetSize.X, CurrentBoxCenter.Y * WidgetSize.Y);
        FVector2D BoxSize = FVector2D(CurrentBoxSize.X * WidgetSize.X, CurrentBoxSize.Y * WidgetSize.Y);

        FVector2D TopLeft = BoxCenter - BoxSize * 0.5f;
        FVector2D BottomRight = BoxCenter + BoxSize * 0.5f;

        // Draw box as 4 lines
        TArray<FVector2D> BoxLines;

        // Top edge
        BoxLines.Empty();
        BoxLines.Add(TopLeft);
        BoxLines.Add(FVector2D(BottomRight.X, TopLeft.Y));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            BoxLines, ESlateDrawEffect::None, BoundingBoxColor, true, BoundingBoxThickness);

        // Bottom edge
        BoxLines.Empty();
        BoxLines.Add(FVector2D(TopLeft.X, BottomRight.Y));
        BoxLines.Add(BottomRight);
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            BoxLines, ESlateDrawEffect::None, BoundingBoxColor, true, BoundingBoxThickness);

        // Left edge
        BoxLines.Empty();
        BoxLines.Add(TopLeft);
        BoxLines.Add(FVector2D(TopLeft.X, BottomRight.Y));
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            BoxLines, ESlateDrawEffect::None, BoundingBoxColor, true, BoundingBoxThickness);

        // Right edge
        BoxLines.Empty();
        BoxLines.Add(FVector2D(BottomRight.X, TopLeft.Y));
        BoxLines.Add(BottomRight);
        FSlateDrawElement::MakeLines(OutDrawElements, MaxLayerId++, AllottedGeometry.ToPaintGeometry(),
            BoxLines, ESlateDrawEffect::None, BoundingBoxColor, true, BoundingBoxThickness);
    }

    // Draw skeleton connections
    if (bShowSkeleton)
    {
        // COCO skeleton connections for 17 keypoints
        const TArray<TPair<int32, int32>> Connections = {
            // Face
            {0, 1}, {0, 2}, {1, 3}, {2, 4},  // Nose to eyes to ears

            // Upper body
            {5, 6},   // Shoulders
            {5, 7}, {7, 9},   // Left arm
            {6, 8}, {8, 10},  // Right arm
            {5, 11}, {6, 12}, // Shoulders to hips

            // Lower body
            {11, 12}, // Hips
            {11, 13}, {13, 15}, // Left leg
            {12, 14}, {14, 16}  // Right leg
        };

        for (const TPair<int32, int32>& Connection : Connections)
        {
            int32 Idx1 = Connection.Key;
            int32 Idx2 = Connection.Value;

            if (Idx1 >= CurrentKeypoints.Num() || Idx2 >= CurrentKeypoints.Num())
            {
                continue;
            }

            // Check visibility (stored in Z)
            if (CurrentKeypointVisibility[Idx1] < 0.5f || CurrentKeypointVisibility[Idx2] < 0.5f)
            {
                continue;
            }

            FVector2D Pos1 = FVector2D(CurrentKeypoints[Idx1].X * WidgetSize.X, CurrentKeypoints[Idx1].Y * WidgetSize.Y);
            FVector2D Pos2 = FVector2D(CurrentKeypoints[Idx2].X * WidgetSize.X, CurrentKeypoints[Idx2].Y * WidgetSize.Y);

            TArray<FVector2D> Line;
            Line.Add(Pos1);
            Line.Add(Pos2);

            FSlateDrawElement::MakeLines(
                OutDrawElements,
                MaxLayerId++,
                AllottedGeometry.ToPaintGeometry(),
                Line,
                ESlateDrawEffect::None,
                SkeletonColor,
                true,
                SkeletonThickness
            );
        }
    }

    // Draw keypoints
    if (bShowKeypoints)
    {
        for (int32 i = 0; i < CurrentKeypoints.Num(); ++i)
        {
            // Skip if low visibility
            if (CurrentKeypointVisibility[i] < 0.5f)
            {
                continue;
            }

            FVector2D ScreenPos = FVector2D(
                CurrentKeypoints[i].X * WidgetSize.X,
                CurrentKeypoints[i].Y * WidgetSize.Y
            );

            // Color coding by body part
            FLinearColor PointColor = KeypointColor;
            if (i == 0) PointColor = FLinearColor::Red;  // Nose
            else if (i >= 1 && i <= 4) PointColor = FLinearColor(1.0f, 0.5f, 0.0f);  // Face (orange)
            else if (i >= 5 && i <= 10) PointColor = FLinearColor::Green;  // Arms
            else if (i >= 11 && i <= 16) PointColor = FLinearColor::Blue;  // Legs

            // Draw keypoint circle
            FVector2D Position = ScreenPos - FVector2D(KeypointSize / 2.0f);
            FVector2D Size = FVector2D(KeypointSize);

            FSlateDrawElement::MakeBox(
                OutDrawElements,
                MaxLayerId++,
                AllottedGeometry.ToPaintGeometry(
                    FVector2f(Size),
                    FSlateLayoutTransform(FVector2f(Position))
                ),
                FCoreStyle::Get().GetBrush(TEXT("WhiteBrush")),
                ESlateDrawEffect::None,
                PointColor
            );
        }
    }

    // Draw confidence
    if (bShowConfidence)
    {
        FString ConfidenceText = FString::Printf(TEXT("YOLO Pose: %.0f%%"), CurrentConfidence * 100.0f);
        FVector2D TextPosition = FVector2D(10, 10);
        FVector2D TextSize = FVector2D(200, 30);

        FSlateDrawElement::MakeText(
            OutDrawElements,
            MaxLayerId++,
            AllottedGeometry.ToPaintGeometry(
                FVector2f(TextSize),
                FSlateLayoutTransform(FVector2f(TextPosition))
            ),
            FText::FromString(ConfidenceText),
            FCoreStyle::GetDefaultFontStyle("Regular", 14),
            ESlateDrawEffect::None,
            FLinearColor::Green
        );
    }

    return MaxLayerId;
}

void UPoseVisualizationWidget::UpdatePoseData(const TArray<FVector>& JointPositions, float Confidence)
{
    // YOLO layout:
    // [0] = Box center (X, Y, 0)
    // [1] = Box size (W, H, 0)
    // [2-18] = 17 keypoints (X, Y, Visibility)

    if (JointPositions.Num() < 19 || Confidence <= 0.0f)
    {
        ClearDetection();
        return;
    }

    // Store previous for smoothing
    FVector2D PrevBoxCenter = CurrentBoxCenter;
    FVector2D PrevBoxSize = CurrentBoxSize;
    TArray<FVector2D> PrevKeypoints = CurrentKeypoints;

    // Extract bounding box
    float BoxCenterX = JointPositions[0].X;
    float BoxCenterY = JointPositions[0].Y;
    float BoxW = JointPositions[1].X;
    float BoxH = JointPositions[1].Y;

    // Apply coordinate transformations
    if (bFlipHorizontal)
    {
        BoxCenterX = 1.0f - BoxCenterX;
    }
    if (bFlipVertical)
    {
        BoxCenterY = 1.0f - BoxCenterY;
    }

    CurrentBoxCenter = FVector2D(BoxCenterX, BoxCenterY);
    CurrentBoxSize = FVector2D(BoxW, BoxH);

    // Extract 17 keypoints
    CurrentKeypoints.Empty();
    CurrentKeypointVisibility.Empty();

    for (int32 i = 2; i < 19; ++i)
    {
        float KpX = JointPositions[i].X;
        float KpY = JointPositions[i].Y;
        float KpVis = JointPositions[i].Z;

        if (bFlipHorizontal)
        {
            KpX = 1.0f - KpX;
        }
        if (bFlipVertical)
        {
            KpY = 1.0f - KpY;
        }

        CurrentKeypoints.Add(FVector2D(KpX, KpY));
        CurrentKeypointVisibility.Add(KpVis);
    }

    CurrentConfidence = Confidence;
    bHasValidDetection = true;

    // Apply smoothing
    if (bEnableSmoothing && PrevKeypoints.Num() == CurrentKeypoints.Num())
    {
        CurrentBoxCenter = FMath::Lerp(PrevBoxCenter, CurrentBoxCenter, SmoothingFactor);
        CurrentBoxSize = FMath::Lerp(PrevBoxSize, CurrentBoxSize, SmoothingFactor);

        for (int32 i = 0; i < CurrentKeypoints.Num(); ++i)
        {
            CurrentKeypoints[i] = FMath::Lerp(PrevKeypoints[i], CurrentKeypoints[i], SmoothingFactor);
        }
    }

    // Debug logging
    static bool bLoggedOnce = false;
    if (!bLoggedOnce)
    {
        UE_LOG(LogTemp, Warning, TEXT("=== YOLO POSE WIDGET ==="));
        UE_LOG(LogTemp, Warning, TEXT("Box Center: (%.3f, %.3f)"), CurrentBoxCenter.X, CurrentBoxCenter.Y);
        UE_LOG(LogTemp, Warning, TEXT("Box Size: (%.3f, %.3f)"), CurrentBoxSize.X, CurrentBoxSize.Y);
        UE_LOG(LogTemp, Warning, TEXT("Keypoints: %d"), CurrentKeypoints.Num());
        UE_LOG(LogTemp, Warning, TEXT("Confidence: %.1f%%"), CurrentConfidence * 100.0f);
        bLoggedOnce = true;
    }
}

void UPoseVisualizationWidget::UpdateDetection(const TArray<FVector>& JointPositions, float Confidence)
{
    // Alias - calls UpdatePoseData
    UpdatePoseData(JointPositions, Confidence);
}

void UPoseVisualizationWidget::ClearDetection()
{
    bHasValidDetection = false;
    CurrentConfidence = 0.0f;
    CurrentKeypoints.Empty();
    CurrentKeypointVisibility.Empty();
}