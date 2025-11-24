// Copyright Epic Games, Inc. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class AIRuntime : ModuleRules
{
	public AIRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		PublicIncludePaths.AddRange(
			new string[] {
                    Path.Combine(ModuleDirectory, "ThirdParty/qairt/include/SNPE"),
                    Path.Combine(ModuleDirectory, "SNPEChaining"),
            }
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                
                // UMG Dependencies (REQUIRED for UUserWidget)
                "UMG",
                "Slate",
                "SlateCore",
                
                // Image processing
                "ImageWrapper",
                
                // Additional modules as needed
                "RHI",
                "RenderCore"
            }
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);

        if (Target.Platform == UnrealTargetPlatform.Android)
        {
            PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "Launch",
            }
            );
            PublicAdditionalLibraries.AddRange(
            new string[] {
                        Path.Combine(ModuleDirectory, "ThirdParty/qairt/lib/Android/libSNPE.so"),
                        Path.Combine(ModuleDirectory, "ThirdParty/qairt/lib/Android/libsnpe-android.so"),
                        Path.Combine(ModuleDirectory, "ThirdParty/qairt/lib/Android/libSnpeHtpPrepare.so"),
                        //Path.Combine(ModuleDirectory, "ThirdParty/qairt/lib/Android/libSnpeHtpV68Skel.so"),
                        //Path.Combine(ModuleDirectory, "ThirdParty/qairt/lib/Android/libSnpeHtpV79Stub.so"),
            }
            );

            string BuildPath = Path.Combine(ModuleDirectory, "QAIRT_APL.xml");
            System.Console.WriteLine(">> QAIRT_APL.xml path: " + BuildPath);
            AdditionalPropertiesForReceipt.Add("AndroidPlugin", BuildPath);
        }
        bEnableExceptions = true;
    }
}
