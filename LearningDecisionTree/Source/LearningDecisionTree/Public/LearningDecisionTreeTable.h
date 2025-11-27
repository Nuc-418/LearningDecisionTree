#pragma once

#include "CoreMinimal.h"
#include "LearningDecisionTreeTable.generated.h"

/**
 * A table structure for Learning Decision Tree.
 * Represents a dataset where columns are features and rows are instances.
 */
USTRUCT(BlueprintType)
struct FLearningDecisionTreeTable
{
	GENERATED_BODY()

public:
	// Maps Column Name -> List of Values (Rows)
	// Note: Nested Containers (TMap<FName, TArray<>>) are not supported in Blueprints (BlueprintReadWrite)
	// We remove BlueprintReadWrite to avoid UHT error. Access should be via functions.
	UPROPERTY(EditAnywhere, Category = "LearningDecisionTree")
	TMap<FName, TArray<int32>> TableData;

	// Keeps track of column order
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	TArray<FName> ColumnNames;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	int32 TotalRows = 0;

	FLearningDecisionTreeTable();

	// Helpers
	int32 GetTableRowCount() const;
	int32 GetTotalRowCount() const;

	int32 GetStateCount(const FName& Column, int32 State);
	int32 GetStateCount(int32 ColumnIndex, int32 State);

	TArray<int32> GetColumnStates(const FName& Column);
	TArray<int32> GetColumnStates(int32 ColumnIndex);

	int32 GetNumberOfStates(const FName& Column);
	int32 GetNumberOfStates(int32 ColumnIndex);

	FName GetColumnName(int32 ColumnIndex) const;

	bool AddColumn(const FName& Name);
	bool AddRow(const TArray<int32>& Row);

	bool RemoveRow(int32 RowIndex);
	bool RemoveColumn(const FName& Column);
	bool RemoveColumn(int32 ColumnIndex);

	float IndividualStateProbability(const FName& Column, int32 State);
	float IndividualStateProbability(int32 ColumnIndex, int32 State);

	FLearningDecisionTreeTable FilterTableByState(const FName& Column, int32 State);
	FLearningDecisionTreeTable FilterTableByState(int32 ColumnIndex, int32 State);

	void RefreshTable();
	void DebugTable();

	// Helper to get duplicate counts for a row
	int32 GetDuplicateCount(int32 RowIndex) const;
};
