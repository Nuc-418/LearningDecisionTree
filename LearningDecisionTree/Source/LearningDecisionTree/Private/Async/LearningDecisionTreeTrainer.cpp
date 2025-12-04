#include "Async/LearningDecisionTreeTrainer.h"
#include "LearningDecisionTreeTable.h"

TSharedPtr<FShadowNode, ESPMode::ThreadSafe> FLearningDecisionTreeTrainer::Train(const FLearningDecisionTreeTable& InTable)
{
	// Ensure table has at least one feature column and an action column (minimum 2 columns)
	if (InTable.ColumnNames.Num() < 2)
	{
		return nullptr;
	}

	// Queue for iterative processing (Breadth-First Expansion)
	TArray<TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe>> NodesToExplode;

	// Create the root node containing the full data
	TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> RootNode = MakeShared<FShadowTableNode, ESPMode::ThreadSafe>(InTable, 0);

	// Wrap the root in a container so it can be replaced by its result (Decision or Action)
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

	// Action column is the last column in our new API
	int32 ActionColumnIndex = Table.ColumnNames.Num() - 1;
	float ActionEntropy = ColumnEntropy(Table, ActionColumnIndex);

	// If Entropy is not 0 (mixed actions) AND we have features to split on
	// ColumnNames.Num() > 1 means we have at least 1 Feature + 1 Action column
	if (ActionEntropy != 0 && Table.ColumnNames.Num() > 1)
	{
		int32 BestColumn = IndexBestInfoGainColumn(Table);

		// If valid split found (BestColumn != -1 means we found a column with positive info gain)
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

			// Replace the current TableNode with the new DecisionNode in the Parent's list
			if (Node->ParentListReference && Node->ParentListReference->IsValidIndex(Node->ThisNodeIndex))
			{
				(*Node->ParentListReference)[Node->ThisNodeIndex] = DecisionNode;
			}
		}
		else
		{
			// Fallback to ActionNode if no split provides positive information gain
			// This prevents infinite loops with inconsistent data
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

float FLearningDecisionTreeTrainer::ColumnEntropy(const FLearningDecisionTreeTable& Table, int32 Column)
{
	double Entropy = 0;
	TArray<int32> States = const_cast<FLearningDecisionTreeTable&>(Table).GetColumnStates(Column);

	for (int32 State : States)
	{
		float Prob = const_cast<FLearningDecisionTreeTable&>(Table).IndividualStateProbability(Column, State);
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

float FLearningDecisionTreeTrainer::InfoGain(const FLearningDecisionTreeTable& Table, int32 Column)
{
	// Cast away const for method calls (table methods should be const but aren't in our implementation)
	FLearningDecisionTreeTable& MutableTable = const_cast<FLearningDecisionTreeTable&>(Table);

	int32 ActionColumn = Table.ColumnNames.Num() - 1;
	float TotalEntropy = ColumnEntropy(Table, ActionColumn);

	TArray<int32> ColumnStates = MutableTable.GetColumnStates(Column);
	TArray<int32> ActionStates = MutableTable.GetColumnStates(ActionColumn);

	int32 NumActionStates = MutableTable.GetNumberOfStates(ActionColumn);

	for (int32 State : ColumnStates)
	{
		TArray<int32> ActionsCount;
		ActionsCount.Init(0, NumActionStates);

		// Access raw data
		const TArray<int32>& ColData = Table.TableData[Table.ColumnNames[Column]];
		const TArray<int32>& ActionData = Table.TableData[Table.ColumnNames[ActionColumn]];

		for (int32 Row = 0; Row < ColData.Num(); Row++)
		{
			if (ColData[Row] == State)
			{
				int32 ActionVal = ActionData[Row];
				int32 ActionIndex = ActionStates.Find(ActionVal);
				if (ActionIndex != INDEX_NONE)
				{
					ActionsCount[ActionIndex] += Table.DuplicateCounts[Row];
				}
			}
		}

		TotalEntropy -= MutableTable.IndividualStateProbability(Column, State) * ArrayEntropy(ActionsCount, MutableTable.GetStateCount(Column, State));
	}

	return TotalEntropy;
}

int32 FLearningDecisionTreeTrainer::IndexBestInfoGainColumn(const FLearningDecisionTreeTable& Table)
{
	float BestGain = 0.0f; // Initialize to 0 so we only pick columns with positive gain
	int32 BestColumn = -1;

	if (Table.ColumnNames.Num() > 0)
	{
		int32 ActionColumn = Table.ColumnNames.Num() - 1;

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
