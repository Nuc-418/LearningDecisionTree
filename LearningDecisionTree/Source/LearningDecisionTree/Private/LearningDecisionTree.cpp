#include "LearningDecisionTree.h"
#include "LearningDecisionTreeNode.h"
#include "Serialization/ObjectAndNameAsStringProxyArchive.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/MemoryWriter.h"
#include "Serialization/MemoryReader.h"
#include "Async/Async.h"

ULearningDecisionTree::ULearningDecisionTree()
{
}

int32 ULearningDecisionTree::GetColumnCount() const
{
	return Table.TableData.Num();
}

int32 ULearningDecisionTree::GetTableRowCount() const
{
	return Table.GetTableRowCount();
}

int32 ULearningDecisionTree::GetTotalRowCount() const
{
	return Table.GetTotalRowCount();
}

void ULearningDecisionTree::AddColumn(FName ColumnName)
{
	Table.AddColumn(ColumnName);
}

void ULearningDecisionTree::AddRow(const TArray<int32>& Row)
{
	// Enforce max unique rows limit
	if (MaxUniqueRows > 0 && Table.GetTableRowCount() >= MaxUniqueRows)
	{
		// Remove oldest row (index 0)
		Table.RemoveRow(0);
	}

	Table.AddRow(Row);
}

void ULearningDecisionTree::CreateDecisionTree()
{
	// Clear previous tree state
	NodesToExplode.Empty();
	LDTRoot.Empty();

	// Create the initial root TableNode containing the full dataset
	ULearningDecisionTreeTableNode* RootNode = NewObject<ULearningDecisionTreeTableNode>(this);
	LDTRoot.Add(RootNode);

	// Init root node. It adds itself to NodesToExplode queue.
	// We pass LDTRoot as the parent list so the RootNode can eventually replace itself
	// with the final DecisionNode or ActionNode.
	RootNode->Init(Table, NodesToExplode, &LDTRoot, 0);

	// Iteratively process nodes until the queue is empty
	// Note: NodesToExplode grows as TableNodes split into children TableNodes.
	while (NodesToExplode.Num() > 0)
	{
		ULearningDecisionTreeNode* Node = NodesToExplode[0];
		if (Node)
		{
			// ExplodeNode will process the node (split it or make it a leaf)
			// and potentially add new children to NodesToExplode.
			Node->ExplodeNode(NodesToExplode);
		}
		// Remove the processed node from the queue
		NodesToExplode.RemoveAt(0);
	}
}

void ULearningDecisionTree::TrainAsync()
{
	if (bIsTraining)
	{
		UE_LOG(LogTemp, Warning, TEXT("Training already in progress. Ignoring request."));
		return;
	}

	bIsTraining = true;

	// Create a deep copy of the table to pass to the background thread
	// FLearningDecisionTreeTable is a struct with TArrays/TMaps, so assignment operator does deep copy.
	FLearningDecisionTreeTable TableSnapshot = Table;

	// Launch async task
	Async(EAsyncExecution::Thread, [this, TableSnapshot]()
	{
		// This runs in a background thread
		TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowRoot = FLearningDecisionTreeTrainer::Train(TableSnapshot);

		// Schedule completion on the Game Thread
		AsyncTask(ENamedThreads::GameThread, [this, ShadowRoot]()
		{
			OnTrainingComplete(ShadowRoot);
		});
	});
}

void ULearningDecisionTree::OnTrainingComplete(TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowRoot)
{
	bIsTraining = false;

	if (ShadowRoot.IsValid())
	{
		// Convert Shadow Tree to UObject Tree
		LDTRoot.Empty();
		ULearningDecisionTreeNode* NewRoot = ConvertShadowToUObject(ShadowRoot, this);
		if (NewRoot)
		{
			LDTRoot.Add(NewRoot);
		}
		UE_LOG(LogTemp, Log, TEXT("Async training complete. Decision Tree updated."));
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Async training failed: Invalid ShadowRoot."));
	}
}

ULearningDecisionTreeNode* ULearningDecisionTree::ConvertShadowToUObject(TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowNode, UObject* Outer)
{
	if (!ShadowNode.IsValid())
	{
		return nullptr;
	}

	ULearningDecisionTreeNode* ResultNode = nullptr;

	switch (ShadowNode->GetType())
	{
	case FShadowNode::DecisionNode:
	{
		TSharedPtr<FShadowDecisionNode, ESPMode::ThreadSafe> ShadowDNode = StaticCastSharedPtr<FShadowDecisionNode>(ShadowNode);
		ULearningDecisionTreeDecisionNode* DNode = NewObject<ULearningDecisionTreeDecisionNode>(Outer);

		// Copy properties
		DNode->BestInfoGainColumn = ShadowDNode->BestInfoGainColumn;
		DNode->ColumnStates = ShadowDNode->ColumnStates;

		// Recursively convert children
		for (TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ChildShadow : ShadowDNode->NextNodes)
		{
			ULearningDecisionTreeNode* ChildUObject = ConvertShadowToUObject(ChildShadow, Outer);
			// Note: If child is null (shouldn't happen), we might add a null entry or handle error.
			// The original code expects matching indices, so we add it even if null (though it won't be null).
			DNode->Nodes.Add(ChildUObject);
		}

		ResultNode = DNode;
		break;
	}
	case FShadowNode::ActionNode:
	{
		TSharedPtr<FShadowActionNode, ESPMode::ThreadSafe> ShadowANode = StaticCastSharedPtr<FShadowActionNode>(ShadowNode);
		ULearningDecisionTreeActionNode* ANode = NewObject<ULearningDecisionTreeActionNode>(Outer);

		// Copy properties
		ANode->ActionNames = ShadowANode->ActionNames;
		ANode->ActionCount = ShadowANode->ActionCounts;

		ResultNode = ANode;
		break;
	}
	case FShadowNode::TableNode:
	{
		// Should not happen in a fully built tree, as TableNodes are temporary during construction.
		// However, if the tree stopped early, we might see one.
		TSharedPtr<FShadowTableNode, ESPMode::ThreadSafe> ShadowTNode = StaticCastSharedPtr<FShadowTableNode>(ShadowNode);
		ULearningDecisionTreeTableNode* TNode = NewObject<ULearningDecisionTreeTableNode>(Outer);

		// TableNodes don't store much persistent state other than the table itself,
		// but in the final tree, they are usually replaced.
		// If we return one, it's a leaf that hasn't been exploded.
		// TNode->Init(...) requires args we don't have easily here, and TableNodes aren't useful for Eval().
		// We'll return it, but it might not be functional for prediction.
		ResultNode = TNode;
		break;
	}
	}

	return ResultNode;
}

void ULearningDecisionTree::RefreshStates(const TArray<int32>& Row)
{
	RowRealTimeStates = Row;
}

int32 ULearningDecisionTree::Eval()
{
	if (LDTRoot.Num() > 0 && LDTRoot[0])
	{
		return LDTRoot[0]->Eval(RowRealTimeStates);
	}
	return -1;
}

void ULearningDecisionTree::DebugTable()
{
	Table.DebugTable();
}

// ============================================================================
// Serialization
// ============================================================================

void ULearningDecisionTree::SaveTable(FString FolderPath, FString FileName)
{
	// Serialize Table struct
	FString FullPath = FPaths::Combine(FolderPath, FileName + TEXT(".dat"));

	TArray<uint8> Bytes;
	FMemoryWriter MemoryWriter(Bytes, true);
	FLearningDecisionTreeTable& TableRef = Table;

	// Manual serialization to ensure stability and control over format

	MemoryWriter << TableRef.TotalRows;

	// Serialize ColumnNames as strings for safety/portability
	int32 NumCols = TableRef.ColumnNames.Num();
	MemoryWriter << NumCols;
	for (const FName& Name : TableRef.ColumnNames)
	{
		FString NameStr = Name.ToString();
		MemoryWriter << NameStr;
	}

	// Serialize TableData Map manually
	int32 NumColumns = TableRef.TableData.Num();
	MemoryWriter << NumColumns;
	for (auto& Pair : TableRef.TableData)
	{
		FName Key = Pair.Key;
		TArray<int32> Value = Pair.Value;
		FString KeyStr = Key.ToString();
		MemoryWriter << KeyStr;
		MemoryWriter << Value;
	}

	FFileHelper::SaveArrayToFile(Bytes, *FullPath);
}

void ULearningDecisionTree::LoadTable(FString FolderPath, FString FileName)
{
	FString FullPath = FPaths::Combine(FolderPath, FileName + TEXT(".dat"));
	TArray<uint8> Bytes;

	if (FFileHelper::LoadFileToArray(Bytes, *FullPath))
	{
		FMemoryReader MemoryReader(Bytes, true);

		Table.TableData.Empty();
		Table.ColumnNames.Empty();
		Table.TotalRows = 0;

		MemoryReader << Table.TotalRows;

		// Deserialize ColumnNames
		int32 NumColsInNames = 0;
		MemoryReader << NumColsInNames;
		for (int32 i = 0; i < NumColsInNames; i++)
		{
			FString NameStr;
			MemoryReader << NameStr;
			Table.ColumnNames.Add(FName(*NameStr));
		}

		// Deserialize TableData
		int32 NumColumns = 0;
		MemoryReader << NumColumns;
		for (int32 i = 0; i < NumColumns; i++)
		{
			FString KeyStr;
			TArray<int32> Value;
			MemoryReader << KeyStr;
			MemoryReader << Value;
			Table.TableData.Add(FName(*KeyStr), Value);
		}
	}
}

void ULearningDecisionTree::SaveDecisionTree(FString FolderPath, FString FileName)
{
	// Saving the UObject tree requires handling pointers and polymorphism.

	FString FullPath = FPaths::Combine(FolderPath, FileName + TEXT(".tree"));

	TArray<uint8> Bytes;
	FMemoryWriter MemoryWriter(Bytes, true);

	// Use a Proxy Archive to handle UProperties of the nodes
	FObjectAndNameAsStringProxyArchive Ar(MemoryWriter, true);
	Ar.ArIsSaveGame = false; // Serialize all properties

	// Recursively serialize the tree starting from the root list
	SerializeNodes(Ar, LDTRoot);

	FFileHelper::SaveArrayToFile(Bytes, *FullPath);
}

void ULearningDecisionTree::LoadDecisionTree(FString FolderPath, FString FileName)
{
	FString FullPath = FPaths::Combine(FolderPath, FileName + TEXT(".tree"));
	TArray<uint8> Bytes;

	if (FFileHelper::LoadFileToArray(Bytes, *FullPath))
	{
		FMemoryReader MemoryReader(Bytes, true);
		FObjectAndNameAsStringProxyArchive Ar(MemoryReader, true);
		Ar.ArIsSaveGame = false;

		// Deserialize recursively, reconstructing the UObject graph
		DeserializeNodes(Ar, LDTRoot, this);
	}
}

// Helpers for manual polymorphic serialization of the node tree

static void SerializeSingleNode(FArchive& Ar, ULearningDecisionTreeNode* Node);
static ULearningDecisionTreeNode* DeserializeSingleNode(FArchive& Ar, UObject* Outer);

void SerializeNodes(FArchive& Ar, TArray<ULearningDecisionTreeNode*>& Nodes)
{
	int32 Num = Nodes.Num();
	Ar << Num;
	for (ULearningDecisionTreeNode* Node : Nodes)
	{
		SerializeSingleNode(Ar, Node);
	}
}

void DeserializeNodes(FArchive& Ar, TArray<ULearningDecisionTreeNode*>& Nodes, UObject* Outer)
{
	Nodes.Empty();
	int32 Num = 0;
	Ar << Num;
	for (int32 i = 0; i < Num; i++)
	{
		ULearningDecisionTreeNode* Node = DeserializeSingleNode(Ar, Outer);
		if (Node)
		{
			Nodes.Add(Node);
		}
	}
}

static void SerializeSingleNode(FArchive& Ar, ULearningDecisionTreeNode* Node)
{
	// Write a type identifier to handle polymorphism
	// 0: Null, 1: TableNode, 2: DecisionNode, 3: ActionNode
	uint8 NodeType = 0;
	if (!Node)
	{
		Ar << NodeType;
		return;
	}

	if (Node->IsA(ULearningDecisionTreeTableNode::StaticClass())) NodeType = 1;
	else if (Node->IsA(ULearningDecisionTreeDecisionNode::StaticClass())) NodeType = 2;
	else if (Node->IsA(ULearningDecisionTreeActionNode::StaticClass())) NodeType = 3;

	Ar << NodeType;

	// Serialize properties using the archive
	Node->Serialize(Ar);

	// For DecisionNode, we need to recursively serialize its children
	if (NodeType == 2) // DecisionNode
	{
		ULearningDecisionTreeDecisionNode* DNode = Cast<ULearningDecisionTreeDecisionNode>(Node);
		SerializeNodes(Ar, DNode->Nodes);
	}
	// TableNodes and ActionNodes don't have persistent children structure to save
}

static ULearningDecisionTreeNode* DeserializeSingleNode(FArchive& Ar, UObject* Outer)
{
	uint8 NodeType = 0;
	Ar << NodeType;

	ULearningDecisionTreeNode* Node = nullptr;

	// Instantiate the correct class based on type identifier
	if (NodeType == 1) Node = NewObject<ULearningDecisionTreeTableNode>(Outer);
	else if (NodeType == 2) Node = NewObject<ULearningDecisionTreeDecisionNode>(Outer);
	else if (NodeType == 3) Node = NewObject<ULearningDecisionTreeActionNode>(Outer);

	if (Node)
	{
		// Deserialize properties
		Node->Serialize(Ar);

		// Recursively deserialize children for DecisionNodes
		if (NodeType == 2)
		{
			ULearningDecisionTreeDecisionNode* DNode = Cast<ULearningDecisionTreeDecisionNode>(Node);
			DeserializeNodes(Ar, DNode->Nodes, Outer);
		}
	}

	return Node;
}
