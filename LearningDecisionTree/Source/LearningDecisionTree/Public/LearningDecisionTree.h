#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "LearningDecisionTreeTable.h"
#include "LearningDecisionTreeNode.h"
#include "Async/LearningDecisionTreeTrainer.h"
#include "LearningDecisionTree.generated.h"

/**
 * Main class for Learning Decision Tree.
 * Manages the Table and the Decision Tree structure using the ID3 algorithm.
 *
 * Usage:
 * 1. Define columns using AddColumn() - feature columns first, then the Action column LAST.
 *    Example: AddColumn("EnemyNearby"); AddColumn("LowHealth"); AddColumn("Action");
 * 2. Feed training data using AddRow() - values for all columns including action.
 *    Example: AddRow({1, 0, 2}); // Enemy nearby=1, LowHealth=0, Action=2
 * 3. Call CreateDecisionTree() to generate the ID3 tree.
 * 4. Use RefreshStates() with feature values only, then Eval() to predict actions.
 *    Example: RefreshStates({1, 0}); int32 Action = Eval();
 *
 * Note: Duplicate rows are handled automatically - no need to add a duplicates column.
 */
UCLASS(BlueprintType)
class LEARNINGDECISIONTREE_API ULearningDecisionTree : public UObject
{
	GENERATED_BODY()

public:
	ULearningDecisionTree();

	/** The data table storing the training data. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	FLearningDecisionTreeTable Table;

	/**
	 * The Root of the decision tree.
	 * It's stored as an array because the tree building process relies on pointer manipulation within a list.
	 * The actual root node is typically at index 0.
	 */
	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> LDTRoot;

	/** Queue of nodes that need to be processed (exploded) during tree creation. */
	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> NodesToExplode;

	/** The current state of the environment, used for evaluation/prediction. */
	UPROPERTY()
	TArray<int32> RowRealTimeStates;

	/** Maximum number of unique rows allowed in the table. 0 = no limit. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree|Settings")
	int32 MaxUniqueRows = 0;

	// Helpers

	/** Returns the number of columns in the table. */
	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetColumnCount() const;

	/** Returns the number of physical rows in the table. */
	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetTableRowCount() const;

	/** Returns the total number of samples (including duplicates). */
	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	int32 GetTotalRowCount() const;

	// Functions

	/** Adds a new column (feature) to the table. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void AddColumn(FName ColumnName);

	/** Adds a training sample (row) to the table. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void AddRow(const TArray<int32>& Row);

	/**
	 * Generates the decision tree based on the current data in the Table using the ID3 algorithm.
	 * This process consumes the data and builds a node structure in LDTRoot.
	 * WARNING: This runs on the Main Thread and may cause lag for large datasets.
	 */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void CreateDecisionTree();

	/**
	 * Starts the decision tree generation in a background thread.
	 * When finished, the main thread will update the tree structure (LDTRoot).
	 * Safe to call during gameplay without causing frame drops.
	 */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void TrainAsync();

	/** Returns true if an async training task is currently running. */
	UFUNCTION(BlueprintPure, Category = "LearningDecisionTree")
	bool IsTraining() const { return bIsTraining; }

	/** Updates the current state vector used for evaluation. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void RefreshStates(const TArray<int32>& Row);

	// Save/Load
	// Provides custom binary serialization to save the training data and the generated tree model to disk.

	/** Saves the Table data to a binary file. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void SaveTable(FString FolderPath, FString FileName);

	/** Loads Table data from a binary file. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void LoadTable(FString FolderPath, FString FileName);

	/** Saves the generated Decision Tree structure to a binary file. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void SaveDecisionTree(FString FolderPath, FString FileName);

	/** Loads a Decision Tree structure from a binary file. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void LoadDecisionTree(FString FolderPath, FString FileName);

	/**
	 * Evaluates the current state (RowRealTimeStates) against the decision tree.
	 * Returns the predicted Action ID.
	 * Returns -1 if no action found or tree is invalid.
	 */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	int32 Eval();

	/** Prints table debug info to log. */
	UFUNCTION(BlueprintCallable, Category = "LearningDecisionTree")
	void DebugTable();

private:
	/** Flag indicating if async training is in progress. */
	bool bIsTraining = false;

	/** Callback for when the async training is complete. */
	void OnTrainingComplete(TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowRoot);

	/** Recursive helper to convert Shadow Tree back to UObject Tree. */
	ULearningDecisionTreeNode* ConvertShadowToUObject(TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowNode, UObject* Outer);
};
