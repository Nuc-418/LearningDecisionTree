#include "Async/LearningDecisionTreeTrainer.h"
#include "LearningDecisionTreeTable.h"

TSharedPtr<FShadowNode, ESPMode::ThreadSafe> FLearningDecisionTreeTrainer::Train(const FLearningDecisionTreeTable& InTable)
{
	// Ensure table has at least Action and Duplicates columns (minimum 2)
	if (InTable.TableData.Num() < 2)
	{
		return nullptr;
	}

	// Queue for iterative processing (Breadth-First Expansion)
	TArray<TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe>> NodesToExplode;

	// Create the root node
	// The root starts as a TableNode containing the full data.
	TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> RootNode = MakeShared<FShadowTableNode, ESPMode::ThreadSafe>(InTable, 0);

	// We wrap the root in a container so it can be replaced by its result (Decision or Action)
	TArray<TSharedPtr<FShadowNode, ESPMode::ThreadSafe>> RootContainer;
	RootContainer.Add(RootNode);

	RootNode->ParentListReference = &RootContainer;
	NodesToExplode.Add(RootNode);

	// Iteratively process nodes
	while (NodesToExplode.Num() > 0)
	{
		TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> Node = NodesToExplode[0];
		NodesToExplode.RemoveAt(0);

		if (Node.IsValid())
		{
			ExplodeNode(Node, NodesToExplode);
		}
	}

	// The RootContainer[0] now contains the fully built tree (DecisionNode or ActionNode)
	return RootContainer[0];
}

void FLearningDecisionTreeTrainer::ExplodeNode(TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> Node, TArray<TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe>>& NodesToExplode)
{
	FLearningDecisionTreeTable& Table = Node->Table;

	// Calculate Entropy of the Action Column (last column - 1, since last is Duplicates)
	int32 ActionColumnIndex = Table.TableData.Num() - 2;
	float ActionEntropy = ColumnEntropy(Table, ActionColumnIndex);

	// If Entropy is not 0 (meaning mixed actions) AND we have features to split on
	// Note: TableData.Num() > 2 means we have at least 1 Feature + 1 Action + 1 Duplicates column
	if (ActionEntropy != 0 && Table.TableData.Num() > 2)
	{
		int32 BestColumn = IndexBestInfoGainColumn(Table);

		// If valid split found
		if (BestColumn != -1)
		{
			TArray<int32> StateNames = Table.GetColumnStates(BestColumn);
			TSharedPtr<FShadowDecisionNode, ESPMode::ThreadSafe> DecisionNode = MakeShared<FShadowDecisionNode, ESPMode::ThreadSafe>(StateNames, BestColumn);

			// For each state in the best column, create a child TableNode
			for (int32 i = 0; i < StateNames.Num(); i++)
			{
				int32 State = StateNames[i];
				FLearningDecisionTreeTable FilteredTable = Table.FilterTableByState(BestColumn, State);

				// Create the child node
				TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> ChildNode = MakeShared<FShadowTableNode, ESPMode::ThreadSafe>(FilteredTable, i);

				// Link the child to the DecisionNode's NextNodes list
				DecisionNode->NextNodes.Add(ChildNode);

				// Pass the reference to this specific slot in the parent's list
				ChildNode->ParentListReference = &DecisionNode->NextNodes;

				NodesToExplode.Add(ChildNode);
			}

			// Replace the current TableNode (Self) with the new DecisionNode in the Parent's list
			if (Node->ParentListReference && Node->ParentListReference->IsValidIndex(Node->ThisNodeIndex))
			{
				(*Node->ParentListReference)[Node->ThisNodeIndex] = DecisionNode;
			}
		}
		else
		{
			// Fallback to ActionNode if no split is possible
			TArray<int32> StateNames = Table.GetColumnStates(ActionColumnIndex);
			TArray<int32> Counts;
			for (int32 State : StateNames)
			{
				Counts.Add(Table.GetStateCount(ActionColumnIndex, State));
			}

			TSharedPtr<FShadowActionNode, ESPMode::ThreadSafe> ActionNode = MakeShared<FShadowActionNode, ESPMode::ThreadSafe>(StateNames, Counts);

			if (Node->ParentListReference && Node->ParentListReference->IsValidIndex(Node->ThisNodeIndex))
			{
				(*Node->ParentListReference)[Node->ThisNodeIndex] = ActionNode;
			}
		}
	}
	else
	{
		// Pure node or no features left -> ActionNode
		TArray<int32> StateNames = Table.GetColumnStates(ActionColumnIndex);
		TArray<int32> Counts;
		for (int32 State : StateNames)
		{
			Counts.Add(Table.GetStateCount(ActionColumnIndex, State));
		}

		TSharedPtr<FShadowActionNode, ESPMode::ThreadSafe> ActionNode = MakeShared<FShadowActionNode, ESPMode::ThreadSafe>(StateNames, Counts);

		if (Node->ParentListReference && Node->ParentListReference->IsValidIndex(Node->ThisNodeIndex))
		{
			(*Node->ParentListReference)[Node->ThisNodeIndex] = ActionNode;
		}
	}
}

float FLearningDecisionTreeTrainer::ColumnEntropy(FLearningDecisionTreeTable& Table, int32 Column)
{
	double Entropy = 0;
	TArray<int32> States = Table.GetColumnStates(Column);

	for (int32 State : States)
	{
		float Prob = Table.IndividualStateProbability(Column, State);
		if (Prob > 0)
		{
			Entropy -= Prob * FMath::Log2(Prob);
		}
	}
	return (float)Entropy;
}

float FLearningDecisionTreeTrainer::ArrayEntropy(const TArray<int32>& Occurrences, int32 Total)
{
	double Entropy = 0;
	for (int32 Occ : Occurrences)
	{
		float Prob = (float)Occ / Total;
		if (Prob > 0 && Total > 0)
		{
			Entropy -= Prob * FMath::Log2(Prob);
		}
	}
	return (float)Entropy;
}

float FLearningDecisionTreeTrainer::InfoGain(FLearningDecisionTreeTable& Table, int32 Column)
{
	int32 ActionColumn = Table.TableData.Num() - 2;
	float TotalEntropy = ColumnEntropy(Table, ActionColumn);

	TArray<int32> ColumnStates = Table.GetColumnStates(Column);
	TArray<int32> ActionStates = Table.GetColumnStates(ActionColumn);

	int32 NumActionStates = Table.GetNumberOfStates(ActionColumn);

	for (int32 State : ColumnStates)
	{
		TArray<int32> ActionsCount;
		ActionsCount.Init(0, NumActionStates);

		// Optimization: Access raw data
		const TArray<int32>& ColData = Table.TableData[Table.GetColumnName(Column)].Data; // Access .Data
		const TArray<int32>& ActionData = Table.TableData[Table.GetColumnName(ActionColumn)].Data; // Access .Data
		const TArray<int32>& DupData = Table.TableData[Table.ColumnNames.Last()].Data; // Access .Data

		for (int32 Row = 0; Row < ColData.Num(); Row++)
		{
			if (ColData[Row] == State)
			{
				int32 ActionVal = ActionData[Row];
				// Find index of this action in ActionStates to map to ActionsCount
				int32 ActionIndex = ActionStates.Find(ActionVal);
				if (ActionIndex != INDEX_NONE)
				{
					ActionsCount[ActionIndex] += DupData[Row];
				}
			}
		}

		TotalEntropy -= Table.IndividualStateProbability(Column, State) * ArrayEntropy(ActionsCount, Table.GetStateCount(Column, State));
	}

	return TotalEntropy;
}

int32 FLearningDecisionTreeTrainer::IndexBestInfoGainColumn(FLearningDecisionTreeTable& Table)
{
	float BestGain = 0.0f; // Initialize to 0 so we only pick columns that actually reduce entropy
	int32 BestColumn = -1;

	if (Table.TableData.Num() > 0)
	{
		int32 ActionColumn = Table.TableData.Num() - 2;

		for (int32 Column = 0; Column < ActionColumn; Column++)
		{
			float Gain = InfoGain(Table, Column);
			// Use strict inequality > to avoid infinite splitting on 0 gain columns
			if (Gain > BestGain)
			{
				BestGain = Gain;
				BestColumn = Column;
			}
		}
	}
	return BestColumn;
}
