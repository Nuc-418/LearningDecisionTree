#include "LearningDecisionTreeTable.h"
#include "Misc/ScopeLock.h"

FLearningDecisionTreeTable::FLearningDecisionTreeTable()
{
}

int32 FLearningDecisionTreeTable::GetTableRowCount() const
{
	// Check if we have columns and data, return the count of the first column
	if (ColumnNames.Num() > 0 && TableData.Contains(ColumnNames[0]))
	{
		return TableData[ColumnNames[0]].Data.Num();
	}
	return 0;
}

int32 FLearningDecisionTreeTable::GetTotalRowCount() const
{
	return TotalRows;
}

int32 FLearningDecisionTreeTable::GetStateCount(const FName& Column, int32 State)
{
	if (TableData.Contains(Column))
	{
		int32 StateCount = 0;
		const TArray<int32>& ColumnData = TableData[Column].Data;
		// Last column is always duplicates count
		const TArray<int32>& DuplicatesData = TableData[ColumnNames.Last()].Data;

		for (int32 Row = 0; Row < ColumnData.Num(); Row++)
		{
			if (ColumnData[Row] == State)
			{
				StateCount += DuplicatesData[Row];
			}
		}
		return StateCount;
	}
	return 0;
}

int32 FLearningDecisionTreeTable::GetStateCount(int32 ColumnIndex, int32 State)
{
	// Validate index range, ensuring we don't access the duplicates column as a data column
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num() - 1)
	{
		return GetStateCount(ColumnNames[ColumnIndex], State);
	}
	return 0;
}

TArray<int32> FLearningDecisionTreeTable::GetColumnStates(const FName& Column)
{
	TArray<int32> States;
	if (TableData.Contains(Column))
	{
		const TArray<int32>& ColumnData = TableData[Column].Data;
		for (int32 State : ColumnData)
		{
			if (!States.Contains(State))
			{
				States.Add(State);
			}
		}
	}
	return States;
}

TArray<int32> FLearningDecisionTreeTable::GetColumnStates(int32 ColumnIndex)
{
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num() - 1)
	{
		return GetColumnStates(ColumnNames[ColumnIndex]);
	}
	return TArray<int32>();
}

int32 FLearningDecisionTreeTable::GetNumberOfStates(const FName& Column)
{
	return GetColumnStates(Column).Num();
}

int32 FLearningDecisionTreeTable::GetNumberOfStates(int32 ColumnIndex)
{
	return GetColumnStates(ColumnIndex).Num();
}

FName FLearningDecisionTreeTable::GetColumnName(int32 ColumnIndex) const
{
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num() - 1)
	{
		return ColumnNames[ColumnIndex];
	}
	UE_LOG(LogTemp, Warning, TEXT("Invalid column index: %d"), ColumnIndex);
	return FName();
}

bool FLearningDecisionTreeTable::AddColumn(const FName& Name)
{
	if (TableData.Contains(Name))
	{
		return false;
	}

	TableData.Add(Name, FLearningDecisionTreeColumn());
	ColumnNames.Add(Name);
	return true;
}

bool FLearningDecisionTreeTable::AddRow(const TArray<int32>& Row)
{
	// Row length should be ColumnNames.Num() - 1 because the last column in Table is Duplicates (managed internally)
	// The caller passes values for all feature columns + action column, but NOT the duplicate column.

	if (Row.Num() == TableData.Num() - 1)
	{
		bool bDup = false;
		int32 DupedRow = -1;

		int32 TableRowCount = GetTableRowCount();

		// Check for duplicates
		for (int32 TableRow = 0; TableRow < TableRowCount; TableRow++)
		{
			int32 DupedData = 0;
			for (int32 TableColumn = 0; TableColumn < ColumnNames.Num() - 1; TableColumn++)
			{
				if (TableData[ColumnNames[TableColumn]].Data[TableRow] == Row[TableColumn])
				{
					DupedData++;
				}
			}

			if (DupedData == ColumnNames.Num() - 1)
			{
				DupedRow = TableRow;
				bDup = true;
				break;
			}
		}

		if (bDup)
		{
			// Increment duplicate count for the existing row
			TableData[ColumnNames.Last()].Data[DupedRow]++;
		}
		else
		{
			// Add new row data to all columns
			for (int32 i = 0; i < Row.Num(); i++)
			{
				TableData[ColumnNames[i]].Data.Add(Row[i]);
			}
			// Initialize duplicate count to 1
			TableData[ColumnNames.Last()].Data.Add(1);
		}

		TotalRows++;
		return true;
	}
	return false;
}

bool FLearningDecisionTreeTable::RemoveRow(int32 RowIndex)
{
	if (ColumnNames.Num() > 0 && RowIndex < TableData[ColumnNames[0]].Data.Num())
	{
		// Decrement TotalRows by the number of duplicates in this row
		TotalRows -= TableData[ColumnNames.Last()].Data[RowIndex];

		for (const FName& ColName : ColumnNames)
		{
			TableData[ColName].Data.RemoveAt(RowIndex);
		}
		return true;
	}
	return false;
}

bool FLearningDecisionTreeTable::RemoveColumn(const FName& Column)
{
	if (TableData.Contains(Column))
	{
		TableData.Remove(Column);
		ColumnNames.Remove(Column);
		RefreshTable();
		return true;
	}
	return false;
}

bool FLearningDecisionTreeTable::RemoveColumn(int32 ColumnIndex)
{
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num())
	{
		FName ColName = ColumnNames[ColumnIndex];
		TableData.Remove(ColName);
		ColumnNames.RemoveAt(ColumnIndex);
		RefreshTable();
		return true;
	}
	return false;
}

float FLearningDecisionTreeTable::IndividualStateProbability(const FName& Column, int32 State)
{
	if (TableData.Contains(Column))
	{
		int32 StateDups = 0;
		const TArray<int32>& ColumnData = TableData[Column].Data;
		const TArray<int32>& DuplicatesData = TableData[ColumnNames.Last()].Data;

		for (int32 Row = 0; Row < ColumnData.Num(); Row++)
		{
			if (ColumnData[Row] == State)
			{
				StateDups += DuplicatesData[Row];
			}
		}

		if (TotalRows > 0)
		{
			return (float)StateDups / (float)TotalRows;
		}
	}
	return 0.0f;
}

float FLearningDecisionTreeTable::IndividualStateProbability(int32 ColumnIndex, int32 State)
{
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num() - 1)
	{
		return IndividualStateProbability(ColumnNames[ColumnIndex], State);
	}
	return 0.0f;
}

FLearningDecisionTreeTable FLearningDecisionTreeTable::FilterTableByState(const FName& Column, int32 State)
{
	FLearningDecisionTreeTable NewTable = *this; // Create a deep copy

	if (NewTable.TableData.Contains(Column))
	{
		int32 RowCount = NewTable.GetTableRowCount();
		// Iterate backwards to safely remove rows while iterating
		for (int32 Row = RowCount - 1; Row >= 0; Row--)
		{
			if (NewTable.TableData[Column].Data[Row] != State)
			{
				NewTable.RemoveRow(Row);
			}
		}
		return NewTable;
	}

	UE_LOG(LogTemp, Error, TEXT("Error FilterTableByState: Column %s not found or invalid"), *Column.ToString());
	return FLearningDecisionTreeTable(); // Empty
}

FLearningDecisionTreeTable FLearningDecisionTreeTable::FilterTableByState(int32 ColumnIndex, int32 State)
{
	if (ColumnIndex >= 0 && ColumnIndex < ColumnNames.Num() - 1)
	{
		FName ColName = ColumnNames[ColumnIndex];
		FLearningDecisionTreeTable NewTable = FilterTableByState(ColName, State);
		// Remove the column we just filtered by, as it is no longer entropic
		NewTable.RemoveColumn(ColName);
		return NewTable;
	}

	UE_LOG(LogTemp, Error, TEXT("Error FilterTableByState: Invalid column index"));
	return FLearningDecisionTreeTable();
}


void FLearningDecisionTreeTable::RefreshTable()
{
	// Merge duplicate rows that might have been created after removing a column
	if (TableData.Num() > 1 && GetTableRowCount() > 0)
	{
		int32 RowCount = GetTableRowCount();

		for (int32 SelectedRow = 0; SelectedRow < RowCount; SelectedRow++)
		{
			// Need to re-check count because we might have removed rows
			if (SelectedRow >= GetTableRowCount()) break;

			for (int32 Row = SelectedRow + 1; Row < GetTableRowCount(); /* increment handled inside or loop */)
			{
				int32 DupedStates = 0;
				// Check equality for all columns except the last one (duplicates count)
				for (int32 Column = 0; Column < ColumnNames.Num() - 1; Column++)
				{
					if (TableData[ColumnNames[Column]].Data[SelectedRow] == TableData[ColumnNames[Column]].Data[Row])
					{
						DupedStates++;
					}
				}

				if (DupedStates == ColumnNames.Num() - 1)
				{
					// If rows are identical, merge counts and remove the duplicate
					TableData[ColumnNames.Last()].Data[SelectedRow] += TableData[ColumnNames.Last()].Data[Row];
					RemoveRow(Row);
					// Do not increment Row index, check the same index again (which is now a new row)
				}
				else
				{
					Row++;
				}
			}
		}
	}
}

void FLearningDecisionTreeTable::DebugTable()
{
	FString DebugStr = "";
	for (const FName& Name : ColumnNames)
	{
		DebugStr += Name.ToString() + " ";
	}
	UE_LOG(LogTemp, Log, TEXT("%s"), *DebugStr);

	int32 RowCount = GetTableRowCount();
	if (RowCount > 0)
	{
		for (int32 Row = 0; Row < RowCount; Row++)
		{
			DebugStr = "";
			for (const FName& Name : ColumnNames)
			{
				DebugStr += Name.ToString() + " : " + FString::FromInt(TableData[Name].Data[Row]) + "|";
			}
			UE_LOG(LogTemp, Log, TEXT("%s"), *DebugStr);
		}
	}
}

int32 FLearningDecisionTreeTable::GetDuplicateCount(int32 RowIndex) const
{
	if (ColumnNames.Num() > 0 && RowIndex >= 0 && RowIndex < TableData[ColumnNames.Last()].Data.Num())
	{
		return TableData[ColumnNames.Last()].Data[RowIndex];
	}
	return 0;
}
