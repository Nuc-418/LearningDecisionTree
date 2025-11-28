#pragma once

#include "CoreMinimal.h"
#include "LearningDecisionTreeTable.h"

/**
 * A raw C++ struct representing a node in the decision tree during async training.
 * We cannot use UObjects on background threads, so we use this shadow structure.
 * We use ThreadSafe shared pointers because these are passed between threads.
 */
struct FShadowNode
{
	virtual ~FShadowNode() {}

	// Type identification
	enum ENodeType
	{
		TableNode,
		DecisionNode,
		ActionNode
	};

	virtual ENodeType GetType() const = 0;
};

struct FShadowTableNode : public FShadowNode
{
	FLearningDecisionTreeTable Table;
	int32 ThisNodeIndex;
	TArray<TSharedPtr<FShadowNode, ESPMode::ThreadSafe>>* ParentListReference; // Pointer to the list in the parent that holds this node

	FShadowTableNode(const FLearningDecisionTreeTable& InTable, int32 InIndex)
		: Table(InTable), ThisNodeIndex(InIndex), ParentListReference(nullptr)
	{}

	virtual ENodeType GetType() const override { return TableNode; }
};

struct FShadowDecisionNode : public FShadowNode
{
	TArray<TSharedPtr<FShadowNode, ESPMode::ThreadSafe>> NextNodes;
	TArray<int32> ColumnStates;
	int32 BestInfoGainColumn;

	FShadowDecisionNode(const TArray<int32>& InColumnStates, int32 InBestColumn)
		: ColumnStates(InColumnStates), BestInfoGainColumn(InBestColumn)
	{}

	virtual ENodeType GetType() const override { return DecisionNode; }
};

struct FShadowActionNode : public FShadowNode
{
	TArray<int32> ActionNames;
	TArray<int32> ActionCounts;

	FShadowActionNode(const TArray<int32>& InNames, const TArray<int32>& InCounts)
		: ActionNames(InNames), ActionCounts(InCounts)
	{}

	virtual ENodeType GetType() const override { return ActionNode; }
};

/**
 * Helper class to run the ID3 algorithm on background threads.
 */
class FLearningDecisionTreeTrainer
{
public:
	// Main entry point for training
	static TSharedPtr<FShadowNode, ESPMode::ThreadSafe> Train(const FLearningDecisionTreeTable& InTable);

private:
	static void ExplodeNode(TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> Node, TArray<TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe>>& NodesToExplode);

	static float ColumnEntropy(FLearningDecisionTreeTable& Table, int32 Column);
	static float ArrayEntropy(const TArray<int32>& Occurrences, int32 Total);
	static float InfoGain(FLearningDecisionTreeTable& Table, int32 Column);
	static int32 IndexBestInfoGainColumn(FLearningDecisionTreeTable& Table);
};
