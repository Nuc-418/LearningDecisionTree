#pragma once

#include "CoreMinimal.h"
#include "LearningDecisionTreeTable.generated.h"

/**
 * Wrapper struct for column data arrays.
 * Required because TMap<FName, TArray<int32>> is not supported as a UPROPERTY directly.
 */
USTRUCT(BlueprintType)
struct FLearningDecisionTreeColumn
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TArray<int32> Data;
};

/**
 * A table structure for Learning Decision Tree.
 * Represents a dataset where columns are features and rows are instances.
 * This structure holds the training data for the ID3 algorithm.
 */
USTRUCT(BlueprintType)
struct FLearningDecisionTreeTable
{
	GENERATED_BODY()

public:
	/**
	 * Maps Column Name -> Column Data (wrapped in struct for UHT compatibility).
	 * Stores the actual data of the table.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	TMap<FName, FLearningDecisionTreeColumn> TableData;

	/** Keeps track of column order. The LAST column is always the Action column. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	TArray<FName> ColumnNames;

	/**
	 * Internal array tracking duplicate counts for each physical row.
	 * This is managed automatically - users don't need to add a duplicates column.
	 */
	UPROPERTY()
	TArray<int32> DuplicateCounts;

	/** Total number of logical rows (accounting for duplicates). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LearningDecisionTree")
	int32 TotalRows = 0;

	FLearningDecisionTreeTable();

	/** Returns the number of physical rows in the table data arrays. */
	int32 GetTableRowCount() const;

	/** Returns the total number of rows, including duplicates count. */
	int32 GetTotalRowCount() const;

	/** Gets the count of a specific state (value) in a column, accounting for row duplicates. */
	int32 GetStateCount(const FName& Column, int32 State);
	int32 GetStateCount(int32 ColumnIndex, int32 State);

	/** Returns a list of unique states (values) present in a column. */
	TArray<int32> GetColumnStates(const FName& Column);
	TArray<int32> GetColumnStates(int32 ColumnIndex);

	/** Returns the number of unique states in a column. */
	int32 GetNumberOfStates(const FName& Column);
	int32 GetNumberOfStates(int32 ColumnIndex);

	/** Gets the column name at a specific index. */
	FName GetColumnName(int32 ColumnIndex) const;

	/** Adds a new column to the table. Returns true if successful. */
	bool AddColumn(const FName& Name);

	/** Adds a row of data to the table. Handles duplicate detection and incrementing counts. */
	bool AddRow(const TArray<int32>& Row);

	/** Removes a physical row from the table at the given index. */
	bool RemoveRow(int32 RowIndex);

	/** Removes a column from the table. */
	bool RemoveColumn(const FName& Column);
	bool RemoveColumn(int32 ColumnIndex);

	/** Calculates the probability of a specific state appearing in a column. */
	float IndividualStateProbability(const FName& Column, int32 State);
	float IndividualStateProbability(int32 ColumnIndex, int32 State);

	/**
	 * Creates a new table containing only rows where the specified column has the given state.
	 * Used for splitting the dataset in the ID3 algorithm.
	 */
	FLearningDecisionTreeTable FilterTableByState(const FName& Column, int32 State);
	FLearningDecisionTreeTable FilterTableByState(int32 ColumnIndex, int32 State);

	/** Merges duplicate rows after column removal to keep the table compact. */
	void RefreshTable();

	/** Prints the table contents to the log for debugging purposes. */
	void DebugTable();

	/** Helper to get duplicate counts for a specific physical row index. */
	int32 GetDuplicateCount(int32 RowIndex) const;
};
