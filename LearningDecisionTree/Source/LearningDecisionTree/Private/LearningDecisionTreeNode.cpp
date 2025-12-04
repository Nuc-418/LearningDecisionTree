#include "LearningDecisionTreeNode.h"
#include "LearningDecisionTreeTable.h"

int32 ULearningDecisionTreeNode::Eval(const TArray<int32>& Row)
{
	// Default implementation returns -1 (error/no action)
	return -1;
}

void ULearningDecisionTreeNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	// Default implementation does nothing
}

// ============================================================================
// ULearningDecisionTreeTableNode
// ============================================================================

void ULearningDecisionTreeTableNode::Init(const FLearningDecisionTreeTable& InTable, TArray<ULearningDecisionTreeNode*>& InNodesToExplode, TArray<ULearningDecisionTreeNode*>* InParentList, int32 InIndex)
{
	Table = InTable;
	ParentList = InParentList;
	ThisNodeIndex = InIndex;

	// Add this node to the processing queue to be exploded later
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
			// Entropy = -Sum(p * log2(p))
			Entropy -= StateProb * FMath::Log2(StateProb);
		}
	}
	return (float)Entropy;
}

float ULearningDecisionTreeTableNode::ArrayEntropy(const TArray<int32>& Occurrences, int32 Total)
{
	if (Total == 0)
	{
		return 0.0f;
	}

	double Entropy = 0;
	for (int32 nOcc : Occurrences)
	{
		if (nOcc > 0)
		{
			float StateOcc = (float)nOcc / (float)Total;
			Entropy -= StateOcc * FMath::Log2(StateOcc);
		}
	}
	return (float)Entropy;
}

float ULearningDecisionTreeTableNode::InfoGain(int32 ColumnIndex)
{
	// The target action column is the last column
	int32 ActionColumn = Table.ColumnNames.Num() - 1;
	int32 TableRowCount = Table.GetTableRowCount();

	// Base entropy of the target set
	float Gain = ColumnEntropy(ActionColumn);

	TArray<int32> ColumnStates = Table.GetColumnStates(ColumnIndex);
	TArray<int32> ActionStates = Table.GetColumnStates(ActionColumn);

	// Subtract conditional entropy for each state in the column
	for (int32 State : ColumnStates)
	{
		TArray<int32> ActionsCount;
		ActionsCount.SetNumZeroed(Table.GetNumberOfStates(ActionColumn));

		int32 IndexAction = 0;
		for (int32 Action : ActionStates)
		{
			for (int32 Row = 0; Row < TableRowCount; Row++)
			{
				if (Table.TableData[Table.ColumnNames[ColumnIndex]].Data[Row] == State &&
					Table.TableData[Table.ColumnNames[ActionColumn]].Data[Row] == Action)
				{
					// Add duplicates count to get true frequency
					ActionsCount[IndexAction] += Table.GetDuplicateCount(Row);
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
	int32 BestInfoGainColumn = -1;

	if (Table.ColumnNames.Num() > 0)
	{
		// Iterate all feature columns (excluding Action column which is the last)
		int32 ActionColumn = Table.ColumnNames.Num() - 1;

		for (int32 Column = 0; Column < ActionColumn; Column++)
		{
			float ColumnInfoGain = InfoGain(Column);

			if (BestInfoGain <= ColumnInfoGain)
			{
				BestInfoGain = ColumnInfoGain;
				BestInfoGainColumn = Column;
			}
		}

		if (BestInfoGainColumn == -1) return 0; // Fallback to first column if no gain found
		return BestInfoGainColumn;
	}
	return -1;
}

void ULearningDecisionTreeTableNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	int32 ActionColumn = Table.ColumnNames.Num() - 1;
	float ActionColumnEntropy = ColumnEntropy(ActionColumn);

	// If entropy is not zero (mixed actions) and we have feature columns left to split on
	// (more than just the action column)
	if (ActionColumnEntropy != 0 && Table.ColumnNames.Num() > 1)
	{
		// Find best column to split
		int32 BestCol = IndexBestInfoGainColumn();
		TArray<int32> StateNames = Table.GetColumnStates(BestCol);

		// Create child nodes for each state of the best column
		for (int32 State : StateNames)
		{
			ULearningDecisionTreeTableNode* NewNode = NewObject<ULearningDecisionTreeTableNode>(GetOuter());
			FLearningDecisionTreeTable FilteredTable = Table.FilterTableByState(BestCol, State);
			// Initialize new node and add to processing queue (NodesToExplode)
			// Pass 'NextNodes' as parent list so we can link them to the DecisionNode later
			NewNode->Init(FilteredTable, NodesToExplode, &NextNodes, NextNodes.Num());
			NextNodes.Add(NewNode);
		}

		// Create a DecisionNode to replace this TableNode
		ULearningDecisionTreeDecisionNode* DecisionNode = NewObject<ULearningDecisionTreeDecisionNode>(GetOuter());
		DecisionNode->Init(NextNodes, StateNames, BestCol);

		// CRITICAL: Update children to point to the new DecisionNode's list.
		// Since DecisionNode copies the array NextNodes, the existing children (which are in NodesToExplode)
		// point to the old NextNodes array. We must redirect them to DecisionNode->Nodes.
		for (int32 i = 0; i < DecisionNode->Nodes.Num(); i++)
		{
			if (ULearningDecisionTreeTableNode* ChildNode = Cast<ULearningDecisionTreeTableNode>(DecisionNode->Nodes[i]))
			{
				ChildNode->ParentList = &DecisionNode->Nodes;
				ChildNode->ThisNodeIndex = i;
			}
		}

		// Replace self in the parent list with the new DecisionNode
		if (ParentList && ParentList->IsValidIndex(ThisNodeIndex))
		{
			(*ParentList)[ThisNodeIndex] = DecisionNode;
		}
	}
	else
	{
		// Leaf Node: Create ActionNode
		TArray<int32> IndividualStateCounts;
		TArray<int32> StateNames = Table.GetColumnStates(ActionColumn);

		for (int32 State : StateNames)
		{
			IndividualStateCounts.Add(Table.GetStateCount(ActionColumn, State));
		}

		ULearningDecisionTreeActionNode* ActionNode = NewObject<ULearningDecisionTreeActionNode>(GetOuter());
		ActionNode->Init(StateNames, IndividualStateCounts);

		// Replace self in the parent list with the new ActionNode
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

	// Find which branch to take based on the row value at the split column
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
		// Construct new row without the used column for the next evaluation step
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
	// DecisionNode is a finished node, nothing to explode
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

	// Random selection weighted by counts
	// FMath::RandRange is inclusive on both ends, so use Total - 1
	int32 Rand = FMath::RandRange(0, Total - 1);

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
	// Return an action based on learned probabilities
	int32 Index = RandAction(ActionCounts);
	if (ActionNames.IsValidIndex(Index))
	{
		return ActionNames[Index];
	}
	return -1;
}

void ULearningDecisionTreeActionNode::ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode)
{
	// ActionNode is a leaf node, nothing to explode
}
