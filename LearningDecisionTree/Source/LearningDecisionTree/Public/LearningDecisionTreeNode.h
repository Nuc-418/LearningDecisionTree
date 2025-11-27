#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "LearningDecisionTreeTable.h"
#include "LearningDecisionTreeNode.generated.h"

/**
 * Base Node class for Learning Decision Tree
 */
UCLASS(Abstract, BlueprintType)
class ULearningDecisionTreeNode : public UObject
{
	GENERATED_BODY()

public:
	// Evaluate the row and return an Action ID.
	// We use Action ID (int32) instead of executing a delegate directly,
	// to make it more flexible for Unreal Blueprints/C++.
	// Returns -1 if no action found.
	virtual int32 Eval(const TArray<int32>& Row);

	// Explode the node to grow the tree
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode);
};

/**
 * Table Node: Represents a node that holds data and needs to be split.
 */
UCLASS()
class ULearningDecisionTreeTableNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	UPROPERTY()
	FLearningDecisionTreeTable Table;

	// In C# this was passed as a list, here we might need a reference or pointer to the list managed by the main tree
	// But since ExplodeNode takes the list, maybe we don't need to store it?
	// The C# code stored it in constructor.

	// We need access to the Parent/Main list of nodes to explode if we want to add self to it?
	// Actually C# constructor: `nodesToExplode.Add(this);`

	// We also need access to the List of Nodes where this node resides in the tree structure (nextNodes/lastNodes)
	// The C# structure is a bit weird: `lastNodes[thisNodeIndex] = new ...` replaces itself in the parent list.
	// In Unreal/C++, replacing `this` is tricky.
	// Instead of replacing, maybe the Parent holds a pointer, and we update that pointer?
	// Or we use a wrapper?

	// Let's look at how it's used.
	// `LDTRoot.Add(new TableNode(...))`
	// `lastNodes` seems to be the list containing `this`.

	// I will store a pointer to the array where this node is stored, and the index.
	TArray<ULearningDecisionTreeNode*>* ParentList = nullptr;
	int32 ThisNodeIndex = 0;

	// To avoid pointer issues with TArray reallocation, it might be safer to use a container object or just manage tree carefully.
	// However, `LDTRoot` is a `List<Node>`.

	// Let's implement `Init` function instead of Constructor for passing complex data.

	void Init(const FLearningDecisionTreeTable& InTable, TArray<ULearningDecisionTreeNode*>& InNodesToExplode, TArray<ULearningDecisionTreeNode*>* InParentList, int32 InIndex);

	virtual int32 Eval(const TArray<int32>& Row) override;
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;

private:
	TArray<ULearningDecisionTreeNode*> NextNodes;

	float ColumnEntropy(int32 ColumnIndex);
	float ArrayEntropy(const TArray<int32>& Occurrences, int32 Total);
	float InfoGain(int32 ColumnIndex);
	int32 IndexBestInfoGainColumn();
};

/**
 * Decision Node: Branches based on a column value.
 */
UCLASS()
class ULearningDecisionTreeDecisionNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	UPROPERTY()
	TArray<ULearningDecisionTreeNode*> Nodes;

	UPROPERTY()
	TArray<int32> ColumnStates;

	UPROPERTY()
	int32 BestInfoGainColumn = 0;

	void Init(const TArray<ULearningDecisionTreeNode*>& InNodes, const TArray<int32>& InColumnStates, int32 InBestColumn);

	virtual int32 Eval(const TArray<int32>& Row) override;
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;
};

/**
 * Action Node: Leaf node that returns an action.
 */
UCLASS()
class ULearningDecisionTreeActionNode : public ULearningDecisionTreeNode
{
	GENERATED_BODY()

public:
	UPROPERTY()
	TArray<int32> ActionNames;

	UPROPERTY()
	TArray<int32> ActionCounts;

	void Init(const TArray<int32>& InActionNames, const TArray<int32>& InActionCounts);

	virtual int32 Eval(const TArray<int32>& Row) override;
	virtual void ExplodeNode(TArray<ULearningDecisionTreeNode*>& NodesToExplode) override;

private:
	int32 RandAction(const TArray<int32>& Probs);
};
