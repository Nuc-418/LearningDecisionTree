#include "LearningDecisionTreeNode.h"
#include "LearningDecisionTreeTable.h"

int32 ULearningDecisionTreeNode::Eval(const TArray<int32>& Row)
{
	// Default implementation
	return -1;
}

void ULearningDecisionTreeNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	// Default implementation
}

// ============================================================================
// ULearningDecisionTreeTableNode
// ============================================================================

void ULearningDecisionTreeTableNode::Init(const FLearningDecisionTreeTable& InTable, TArray<ULearningDecisionTreeNode*>& InNodesToExplode, TArray<ULearningDecisionTreeNode*>* InParentList, int32 InIndex)
{
	Table = InTable;
	ParentList = InParentList;
	ThisNodeIndex = InIndex;

	InNodesToExplode.Add(this);
}

float ULearningDecisionTreeTableNode::ColumnEntropy(int32 ColumnIndex)
{
	double Entropy = 0;
	TArray<int32> ColumnStates = Table.GetColumnStates(ColumnIndex);

	for (int32 State : ColumnStates)
	{
		float StateProb = Table.IndividualStateProbability(ColumnIndex, State);
		if (StateProb != 0)
		{
			Entropy -= StateProb * FMath::Log2(StateProb);
		}
	}
	return (float)Entropy;
}

float ULearningDecisionTreeTableNode::ArrayEntropy(const TArray<int32>& Occurrences, int32 Total)
{
	double Entropy = 0;
	for (int32 nOcc : Occurrences)
	{
		float StateOcc = (float)nOcc / (float)Total;
		if (StateOcc != 0 && Total != 0)
		{
			Entropy -= StateOcc * FMath::Log2(StateOcc);
		}
	}
	return (float)Entropy;
}

float ULearningDecisionTreeTableNode::InfoGain(int32 ColumnIndex)
{
	// Action column is the second to last (Last is duplicates)
	// In C#: `int actionColumn = table.tableData.Count - 2;`
	int32 ActionColumn = Table.ColumnNames.Num() - 2;
	int32 TableRowCount = Table.GetTableRowCount();

	float Gain = ColumnEntropy(ActionColumn);

	TArray<int32> ColumnStates = Table.GetColumnStates(ColumnIndex);
	TArray<int32> ActionStates = Table.GetColumnStates(ActionColumn);

	for (int32 State : ColumnStates)
	{
		TArray<int32> ActionsCount;
		ActionsCount.SetNumZeroed(Table.GetNumberOfStates(ActionColumn));

		int32 IndexAction = 0;
		for (int32 Action : ActionStates)
		{
			for (int32 Row = 0; Row < TableRowCount; Row++)
			{
				if (Table.TableData[Table.ColumnNames[ColumnIndex]][Row] == State &&
					Table.TableData[Table.ColumnNames[ActionColumn]][Row] == Action)
				{
					// Add duplicates count
					ActionsCount[IndexAction] += Table.TableData[Table.ColumnNames.Last()][Row];
				}
			}
			IndexAction++;
		}

		Gain -= Table.IndividualStateProbability(ColumnIndex, State) * ArrayEntropy(ActionsCount, Table.GetStateCount(ColumnIndex, State));
	}

	return Gain;
}

int32 ULearningDecisionTreeTableNode::IndexBestInfoGainColumn()
{
	float BestInfoGain = 0;
	int32 BestInfoGainColumn = -1; // Default to -1 or 0? C# defaults to 0 but returns -1 if empty.

	if (Table.ColumnNames.Num() > 0)
	{
		int32 ActionColumn = Table.ColumnNames.Num() - 2;

		for (int32 Column = 0; Column < ActionColumn; Column++)
		{
			float ColumnInfoGain = InfoGain(Column);
			// Changed from <= to < to prioritize first found? No C# uses <=
			if (BestInfoGain <= ColumnInfoGain)
			{
				BestInfoGain = ColumnInfoGain;
				BestInfoGainColumn = Column;
			}
		}
		// If BestInfoGainColumn is still -1 (initially), it means InfoGain was negative? Entropy is positive.
		// If BestInfoGain is 0, we might return 0.
		if (BestInfoGainColumn == -1) return 0; // Fallback
		return BestInfoGainColumn;
	}
	return -1;
}

void ULearningDecisionTreeTableNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	int32 ActionColumn = Table.ColumnNames.Num() - 2;
	float ActionColumnEntropy = ColumnEntropy(ActionColumn);

	// In C#: if (!(actionColumnEntropy == 0) && table.tableData.Count > 2)
	// tableData.Count is number of columns.

	if (ActionColumnEntropy != 0 && Table.ColumnNames.Num() > 2)
	{
		int32 BestCol = IndexBestInfoGainColumn();
		TArray<int32> StateNames = Table.GetColumnStates(BestCol);

		for (int32 State : StateNames)
		{
			ULearningDecisionTreeTableNode* NewNode = NewObject<ULearningDecisionTreeTableNode>(GetOuter());
			FLearningDecisionTreeTable FilteredTable = Table.FilterTableByState(BestCol, State);
			// We pass NextNodes as the parent list for the children
			NewNode->Init(FilteredTable, NodesToExplode, &NextNodes, NextNodes.Num());
			NextNodes.Add(NewNode);
		}

		// Replace self in parent list with DecisionNode
		ULearningDecisionTreeDecisionNode* DecisionNode = NewObject<ULearningDecisionTreeDecisionNode>(GetOuter());
		DecisionNode->Init(NextNodes, StateNames, BestCol);

		// CRITICAL FIX: Update children to point to the new DecisionNode's list.
		// Since DecisionNode copies the array, the existing children (which are in NodesToExplode)
		// point to the old NextNodes array. We must redirect them to DecisionNode->Nodes.
		for (int32 i = 0; i < DecisionNode->Nodes.Num(); i++)
		{
			if (ULearningDecisionTreeTableNode* ChildNode = Cast<ULearningDecisionTreeTableNode>(DecisionNode->Nodes[i]))
			{
				ChildNode->ParentList = &DecisionNode->Nodes;
				ChildNode->ThisNodeIndex = i;
			}
		}

		if (ParentList && ParentList->IsValidIndex(ThisNodeIndex))
		{
			(*ParentList)[ThisNodeIndex] = DecisionNode;
		}
	}
	else
	{
		// Create ActionNode
		TArray<int32> IndividualStateCounts;
		TArray<int32> StateNames = Table.GetColumnStates(ActionColumn);

		for (int32 State : StateNames)
		{
			IndividualStateCounts.Add(Table.GetStateCount(ActionColumn, State));
		}

		ULearningDecisionTreeActionNode* ActionNode = NewObject<ULearningDecisionTreeActionNode>(GetOuter());
		ActionNode->Init(StateNames, IndividualStateCounts);

		if (ParentList && ParentList->IsValidIndex(ThisNodeIndex))
		{
			(*ParentList)[ThisNodeIndex] = ActionNode;
		}
	}
}

int32 ULearningDecisionTreeTableNode::Eval(const TArray<int32>& Row)
{
	UE_LOG(LogTemp, Warning, TEXT("Eval called on TableNode - this should not happen in a fully built tree."));
	return -1;
}


// ============================================================================
// ULearningDecisionTreeDecisionNode
// ============================================================================

void ULearningDecisionTreeDecisionNode::Init(const TArray<ULearningDecisionTreeNode*>& InNodes, const TArray<int32>& InColumnStates, int32 InBestColumn)
{
	Nodes = InNodes;
	ColumnStates = InColumnStates;
	BestInfoGainColumn = InBestColumn;
}

int32 ULearningDecisionTreeDecisionNode::Eval(const TArray<int32>& Row)
{
	int32 SelectedNodeIndex = -1;

	for (int32 i = 0; i < ColumnStates.Num(); i++)
	{
		if (Row.IsValidIndex(BestInfoGainColumn) && ColumnStates[i] == Row[BestInfoGainColumn])
		{
			SelectedNodeIndex = i;
			break;
		}
	}

	if (SelectedNodeIndex > -1 && Nodes.IsValidIndex(SelectedNodeIndex))
	{
		// Construct new row without the used column
		TArray<int32> NewRow;
		for (int32 i = 0; i < Row.Num(); i++)
		{
			if (i != BestInfoGainColumn)
			{
				NewRow.Add(Row[i]);
			}
		}

		return Nodes[SelectedNodeIndex]->Eval(NewRow);
	}

	return -1;
}

void ULearningDecisionTreeDecisionNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	// Do nothing
}


// ============================================================================
// ULearningDecisionTreeActionNode
// ============================================================================

void ULearningDecisionTreeActionNode::Init(const TArray<int32>& InActionNames, const TArray<int32>& InActionCounts)
{
	ActionNames = InActionNames;
	ActionCounts = InActionCounts;
}

int32 ULearningDecisionTreeActionNode::RandAction(const TArray<int32>& Probs)
{
	int32 Total = 0;
	for (int32 Prob : Probs)
	{
		Total += Prob;
	}

	int32 Rand = FMath::RandRange(0, Total); // Inclusive max? FMath::RandRange is [Min, Max] inclusive.

	// C# Logic:
	// rand = Random.Range(0, total + 1); // [0, total]
	// loop...
	// if (rand >= total_accum) ...

	// Let's replicate exact C# logic
	// Note: C# Random.Range(int min, int max) is max-exclusive for integers.
	// WAIT. `Random.Range(0, total + 1)` makes it inclusive of `total`.
	// The loop logic:
	/*
        total = 0;
        int indexObstacle = -1;
        foreach (int prob in probs)
            if (rand >= total)
            {
                total += prob;
                indexObstacle++;
            }
	*/
	// If rand == total (max value), and the loop finishes, indexObstacle will include the last item.

	int32 CurrentTotal = 0;
	int32 IndexObstacle = -1;

	for (int32 Prob : Probs)
	{
		if (Rand >= CurrentTotal)
		{
			CurrentTotal += Prob;
			IndexObstacle++;
		}
	}

	// Clamp to valid range just in case
	if (IndexObstacle < 0) IndexObstacle = 0;
	if (IndexObstacle >= Probs.Num()) IndexObstacle = Probs.Num() - 1;

	return IndexObstacle;
}

int32 ULearningDecisionTreeActionNode::Eval(const TArray<int32>& Row)
{
	int32 Index = RandAction(ActionCounts);
	if (ActionNames.IsValidIndex(Index))
	{
		return ActionNames[Index];
	}
	return -1;
}

void ULearningDecisionTreeActionNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	// Do nothing
}
