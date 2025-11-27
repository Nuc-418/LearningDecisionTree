#include "LearningDecisionTreeTable.h"
#include "Misc/ScopeLock.h"

FLearningDecisionTreeTable::FLearningDecisionTreeTable()
{
}

int32 FLearningDecisionTreeTable::GetTableRowCount() const
{
	if (ColumnNames.Num() > 0 && TableData.Contains(ColumnNames[0]))
	{
		return TableData[ColumnNames[0]].Num();
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
		const TArray<int32>& ColumnData = TableData[Column];
		const TArray<int32>& DuplicatesData = TableData[ColumnNames.Last()]; // Last column is duplicates count

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
		const TArray<int32>& ColumnData = TableData[Column];
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

	TableData.Add(Name, TArray<int32>());
	ColumnNames.Add(Name);
	return true;
}

bool FLearningDecisionTreeTable::AddRow(const TArray<int32>& Row)
{
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
				if (TableData[ColumnNames[TableColumn]][TableRow] == Row[TableColumn])
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
			// Increment duplicate count
			TableData[ColumnNames.Last()][DupedRow]++;
		}
		else
		{
			// Add new row
			for (int32 i = 0; i < Row.Num(); i++)
			{
				TableData[ColumnNames[i]].Add(Row[i]);
			}
			// Initialize duplicate count to 1
			TableData[ColumnNames.Last()].Add(1);
		}

		TotalRows++;
		return true;
	}
	return false;
}

bool FLearningDecisionTreeTable::RemoveRow(int32 RowIndex)
{
	if (ColumnNames.Num() > 0 && RowIndex < TableData[ColumnNames[0]].Num())
	{
		// Decrement TotalRows by the number of duplicates in this row
		TotalRows -= TableData[ColumnNames.Last()][RowIndex];

		for (const FName& ColName : ColumnNames)
		{
			TableData[ColName].RemoveAt(RowIndex);
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
		const TArray<int32>& ColumnData = TableData[Column];
		const TArray<int32>& DuplicatesData = TableData[ColumnNames.Last()];

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
	FLearningDecisionTreeTable NewTable = *this; // Copy

	if (NewTable.TableData.Contains(Column))
	{
		int32 RowCount = NewTable.GetTableRowCount();
		// It's safer to iterate backwards when removing
		for (int32 Row = RowCount - 1; Row >= 0; Row--)
		{
			if (NewTable.TableData[Column][Row] != State)
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
		NewTable.RemoveColumn(ColName);
		return NewTable;
	}

	UE_LOG(LogTemp, Error, TEXT("Error FilterTableByState: Invalid column index"));
	return FLearningDecisionTreeTable();
}


void FLearningDecisionTreeTable::RefreshTable()
{
	if (TableData.Num() > 1 && GetTableRowCount() > 0)
	{
		int32 RowCount = GetTableRowCount();

		for (int32 SelectedRow = 0; SelectedRow < RowCount; SelectedRow++)
		{
			if (SelectedRow >= GetTableRowCount()) break;

			for (int32 Row = SelectedRow + 1; Row < GetTableRowCount(); /* increment handled inside or loop */)
			{
				int32 DupedStates = 0;
				// Check all columns except the last one (duplicates count)
				for (int32 Column = 0; Column < ColumnNames.Num() - 1; Column++)
				{
					if (TableData[ColumnNames[Column]][SelectedRow] == TableData[ColumnNames[Column]][Row])
					{
						DupedStates++;
					}
				}

				if (DupedStates == ColumnNames.Num() - 1)
				{
					// Merge
					TableData[ColumnNames.Last()][SelectedRow] += TableData[ColumnNames.Last()][Row];
					RemoveRow(Row);
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
				DebugStr += Name.ToString() + " : " + FString::FromInt(TableData[Name][Row]) + "|";
			}
			UE_LOG(LogTemp, Log, TEXT("%s"), *DebugStr);
		}
	}
}

int32 FLearningDecisionTreeTable::GetDuplicateCount(int32 RowIndex) const
{
	if (ColumnNames.Num() > 0 && RowIndex >= 0 && RowIndex < TableData[ColumnNames.Last()].Num())
	{
		return TableData[ColumnNames.Last()][RowIndex];
	}
	return 0;
}
