#pragma once

#include "CoreMinimal.h"
#include "LearningDecisionTreeTable.h"

/**
 * A raw C++ struct representing a node in the decision tree during async training.
 * We cannot use UObjects on background threads, so we use this shadow structure.
 * We use ThreadSafe shared pointers because these are passed between threads.
 */
struct LEARNINGDECISIONTREE_API FShadowNode
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

struct LEARNINGDECISIONTREE_API FShadowTableNode : public FShadowNode
{
	FLearningDecisionTreeTable Table;
	int32 ThisNodeIndex;
	TArray<TSharedPtr<FShadowNode, ESPMode::ThreadSafe>>* ParentListReference;

	FShadowTableNode(const FLearningDecisionTreeTable& InTable, int32 InIndex)
		: Table(InTable), ThisNodeIndex(InIndex), ParentListReference(nullptr)
	{}

	virtual ENodeType GetType() const override { return TableNode; }
};

struct LEARNINGDECISIONTREE_API FShadowDecisionNode : public FShadowNode
{
	TArray<TSharedPtr<FShadowNode, ESPMode::ThreadSafe>> NextNodes;
	TArray<int32> ColumnStates;
	int32 BestInfoGainColumn;

	FShadowDecisionNode(const TArray<int32>& InColumnStates, int32 InBestColumn)
		: ColumnStates(InColumnStates), BestInfoGainColumn(InBestColumn)
	{}

	virtual ENodeType GetType() const override { return DecisionNode; }
};

struct LEARNINGDECISIONTREE_API FShadowActionNode : public FShadowNode
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
 * Uses shadow node structures (non-UObject) for thread safety.
 */
class LEARNINGDECISIONTREE_API FLearningDecisionTreeTrainer
{
public:
	/**
	 * Main entry point for training. Can be called from any thread.
	 * @param InTable The training data table
	 * @return The root of the shadow tree (DecisionNode or ActionNode)
	 */
	static TSharedPtr<FShadowNode, ESPMode::ThreadSafe> Train(const FLearningDecisionTreeTable& InTable);

private:
	static void ExplodeNode(TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> Node, TArray<TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe>>& NodesToExplode);

	static float ColumnEntropy(const FLearningDecisionTreeTable& Table, int32 Column);
	static float ArrayEntropy(const TArray<int32>& Occurrences, int32 Total);
	static float InfoGain(const FLearningDecisionTreeTable& Table, int32 Column);
	static int32 IndexBestInfoGainColumn(const FLearningDecisionTreeTable& Table);
};
