#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "LearningDecisionTreeTable.h"
#include "LearningDecisionTreeNode.h"
#include "LearningDecisionTree.generated.h"

/**
 * Main class for Learning Decision Tree.
 * Manages the Table and the Decision Tree structure.
 */
UCLASS(BlueprintType)
class LEARNINGDECISIONTREE_API ULearningDecisionTree : public UObject
{
	GENERATED_BODY()

public:
	ULearningDecisionTree();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	FLearningDecisionTreeTable Table;

	// The Root of the tree. It's a list in C# (`LDTRoot`), but usually it's just one root node?
	// C# `LDTRoot` is `List<Node>`. `LDTRoot[0]` is used for Eval.
	// It seems it's a list because `TableNode` logic modifies this list by index.
	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> LDTRoot;

	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> NodesToExplode;

	UPROPERTY()
	TArray<int32> RowRealTimeStates;

	// Helpers
	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetColumnCount() const;

	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetTableRowCount() const;

	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetTotalRowCount() const;

	// Functions
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void AddColumn(FName ColumnName);

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void AddRow(const TArray<int32>& Row);

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void CreateDecisionTree();

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void RefreshStates(const TArray<int32>& Row);

	// Save/Load
	// We can use generic SaveGame or simple file I/O.
	// For C++ implementation, let's provide File I/O for binary data similar to C# implementation
	// or use JSON? Binary is requested by memory but I can do whatever fits Unreal.
	// User asked for C++ port. I'll stick to Binary using FArchive.

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void SaveTable(FString FolderPath, FString FileName);

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void LoadTable(FString FolderPath, FString FileName);

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void SaveDecisionTree(FString FolderPath, FString FileName);

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void LoadDecisionTree(FString FolderPath, FString FileName);

	/**
	 * Evaluates the current state (RowRealTimeStates) and returns the Action ID.
	 * Returns -1 if no action.
	 */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	int32 Eval();

	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void DebugTable();
};
