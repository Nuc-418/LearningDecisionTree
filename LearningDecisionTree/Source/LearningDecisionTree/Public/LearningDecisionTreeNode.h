#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "LearningDecisionTreeTable.h"
#include "LearningDecisionTreeNode.generated.h"

/**
 * Base Node class for Learning Decision Tree.
 * Defines the common interface for all nodes in the tree.
 */
UCLASS(Abstract, BlueprintType)
class ULearningDecisionTreeNode : public UObject
{
	GENERATED_BODY()

public:
	/**
	 * Evaluates the row and returns an Action ID.
	 * We use Action ID (int32) instead of executing a delegate directly,
	 * to make it more flexible for Unreal Blueprints/C++.
	 * @param Row The input data row to evaluate (features).
	 * @return The resulting Action ID, or -1 if no action found.
	 */
	virtual int32 Eval(const TArray<int32>& Row);

	/**
	 * Explodes (processes) the node to grow the tree.
	 * Used during the tree building process.
	 * @param NodesToExplode The list of nodes pending processing.
	 */
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode);
};

/**
 * Table Node: Represents a node that holds a subset of data and needs to be split.
 * This is a temporary node type used during tree construction.
 * It will eventually be replaced by a DecisionNode or ActionNode in the final tree.
 */
UCLASS()
class ULearningDecisionTreeTableNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	/** The data table subset associated with this node. */
	UPROPERTY()
	FLearningDecisionTreeTable Table;

	/** Pointer to the parent's list of nodes, allowing this node to replace itself with a Decision/Action node. */
	TArray<ULearningDecisionTreeNode*>* ParentList = nullptr;

	/** The index of this node in the ParentList. */
	int32 ThisNodeIndex = 0;

	/**
	 * Initializes the TableNode.
	 * @param InTable The data table for this node.
	 * @param InNodesToExplode Reference to the global list of nodes to process.
	 * @param InParentList Pointer to the list containing this node (in parent).
	 * @param InIndex Index of this node in InParentList.
	 */
	void Init(const FLearningDecisionTreeTable& InTable, TArray<ULearningDecisionTreeNode*>& InNodesToExplode, TArray<ULearningDecisionTreeNode*>* InParentList, int32 InIndex);

	virtual int32 Eval(const TArray<int32>& Row) override;

	/**
	 * Performs the ID3 algorithm step:
	 * 1. Calculates entropy and information gain for each column.
	 * 2. Selects the best column to split on.
	 * 3. Creates child nodes (TableNodes) for each state of the selected column.
	 * 4. Replaces itself with a DecisionNode (pointing to children) or ActionNode (leaf) in the parent list.
	 */
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;

private:
	/** Temporary list to hold children nodes before they are moved to the resulting DecisionNode. */
	TArray<ULearningDecisionTreeNode*> NextNodes;

	/** Calculates entropy for a specific column. */
	float ColumnEntropy(int32 ColumnIndex);

	/** Calculates entropy for an array of occurrences. */
	float ArrayEntropy(const TArray<int32>& Occurrences, int32 Total);

	/** Calculates Information Gain for a specific column. */
	float InfoGain(int32 ColumnIndex);

	/** Finds the column index with the highest Information Gain. */
	int32 IndexBestInfoGainColumn();
};

/**
 * Decision Node: Represents a branch in the decision tree based on a feature column.
 */
UCLASS()
class ULearningDecisionTreeDecisionNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	/** List of child nodes corresponding to each state of the split column. */
	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> Nodes;

	/** The states (values) of the column mapping to the child nodes. */
	UPROPERTY()
	TArray<int32> ColumnStates;

	/** The index of the column this node splits on. */
	UPROPERTY()
	int32 BestInfoGainColumn = 0;

	/** Initializes the DecisionNode. */
	void Init(const TArray<ULearningDecisionTreeNode*>& InNodes, const TArray<int32>& InColumnStates, int32 InBestColumn);

	/**
	 * Traverses the tree by matching the value in the row at BestInfoGainColumn
	 * with ColumnStates, and recursively evaluating the corresponding child node.
	 */
	virtual int32 Eval(const TArray<int32>& Row) override;

	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;
};

/**
 * Action Node: Leaf node that returns an action based on probability.
 */
UCLASS()
class ULearningDecisionTreeActionNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	/** Possible action IDs. */
	UPROPERTY()
	TArray<int32> ActionNames;

	/** Counts/weights for each action, determining probability. */
	UPROPERTY()
	TArray<int32> ActionCounts;

	/** Initializes the ActionNode. */
	void Init(const TArray<int32>& InActionNames, const TArray<int32>& InActionCounts);

	/**
	 * Selects an action probabilistically based on ActionCounts and returns it.
	 */
	virtual int32 Eval(const TArray<int32>& Row) override;

	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;

private:
	/** Selects an index based on weighted probability. */
	int32 RandAction(const TArray<int32>& Probs);
};
