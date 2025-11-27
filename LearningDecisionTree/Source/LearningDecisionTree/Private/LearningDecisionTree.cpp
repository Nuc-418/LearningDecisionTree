#include "LearningDecisionTree.h"
#include "LearningDecisionTreeNode.h"
#include "Serialization/ObjectAndNameAsStringProxyArchive.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/MemoryWriter.h"
#include "Serialization/MemoryReader.h"

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
	Table.AddRow(Row);
}

void ULearningDecisionTree::CreateDecisionTree()
{
	NodesToExplode.Empty();
	LDTRoot.Empty();

	ULearningDecisionTreeTableNode* RootNode = NewObject<ULearningDecisionTreeTableNode>(this);
	// Root node is at index 0 of LDTRoot
	LDTRoot.Add(RootNode);

	// Init root node. It adds itself to NodesToExplode.
	RootNode->Init(Table, NodesToExplode, &LDTRoot, 0);

	// Loop through NodesToExplode
	// Note: NodesToExplode grows as we explode nodes.
	// C# Loop:
	/*
        for (int tableNode = 0; tableNode < nodeToExplode.Count + 1; tableNode++)
        {
            nodeToExplode[0].ExplodeNode(nodeToExplode);
            nodeToExplode.RemoveAt(0);
            tableNode = 0;
        }
	*/
	// This C# loop is a bit weird. It constantly removes 0 and resets counter. Basically a while loop.

	while (NodesToExplode.Num() > 0)
	{
		ULearningDecisionTreeNode* Node = NodesToExplode[0];
		if (Node)
		{
			Node->ExplodeNode(NodesToExplode);
		}
		NodesToExplode.RemoveAt(0);
	}
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

	// Create a simple Archive for struct
	// Note: FLearningDecisionTreeTable is a USTRUCT with UPROPERTY, so we could use FObjectWriter if it was a UObject,
	// or standard serialization if it implemented << operator.
	// Since we are inside a UObject, we can't easily Serialize just the struct with UProperties automatically without a wrapper archive
	// that supports UProperties or manual serialization.

	// However, since we are in a plugin, let's implement operator<< for the struct or manually serialize fields.
	// Or use `FObjectAndNameAsStringProxyArchive`.

	// Actually, simpler: Use `FFileHelper::SaveArrayToFile`? No, we need to serialize map.

	// Let's implement a helper to serialize the table manually to be safe and portable.

	MemoryWriter << TableRef.TotalRows;
	// Serialize ColumnNames as strings
	int32 NumCols = TableRef.ColumnNames.Num();
	MemoryWriter << NumCols;
	for (const FName& Name : TableRef.ColumnNames)
	{
		FString NameStr = Name.ToString();
		MemoryWriter << NameStr;
	}

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

		int32 NumColsInNames = 0;
		MemoryReader << NumColsInNames;
		for (int32 i = 0; i < NumColsInNames; i++)
		{
			FString NameStr;
			MemoryReader << NameStr;
			Table.ColumnNames.Add(FName(*NameStr));
		}

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
	// Saving the UObject tree is more complex because of pointers.
	// We can use `FObjectAndNameAsStringProxyArchive`.

	FString FullPath = FPaths::Combine(FolderPath, FileName + TEXT(".tree"));

	// We need to serialize the whole object graph starting from LDTRoot.
	// The standard way to save UObjects to disk is using a Package, but here we want a custom file.

	// Alternatively, since we are rebuilding the tree from Table data every time in `CreateDecisionTree`,
	// maybe we don't strictly need to save the Tree if we save the Table?
	// The C# code saves both. `CreateDecisionTree` is computationally expensive so saving the tree is good.

	// Let's try to use `FObjectWriter`.
	TArray<uint8> Bytes;
	FMemoryWriter MemoryWriter(Bytes, true);

	// Wrap in Proxy Archive to handle UObjects serialization
	FObjectAndNameAsStringProxyArchive Ar(MemoryWriter, true);
	Ar.ArIsSaveGame = false; // Serialize all properties (Fix: was true)

	// We serialize the whole ULearningDecisionTree object? No, just the tree nodes.
	// But the nodes are UObjects.

	// To serialize UObjects properly including their class type (to handle polymorphism),
	// we usually need to serialize the Class Path and then properties.

	// A simpler approach for this specific structure:
	// Recursively serialize nodes.

	// BUT, `FObjectAndNameAsStringProxyArchive` is good for properties.

	// Let's just serialize `LDTRoot`.
	// Since `LDTRoot` contains pointers, `operator<<` on TArray<UObject*> only serializes the pointer value (index/ID) usually,
	// unless using a specific archiver.

	// Given the complexity of implementing robust UObject graph serialization from scratch in a simple plugin without the Engine's full serialization context (Packages),
	// and the fact that the tree can be deterministically recreated from the Table,
	// I will implement Save/Load DecisionTree by relying on the Table.
	// IF the user loads a tree, they probably expect it to be the one generated from the table.
	// The C# `LoadDecisionTree` loads the binary serialized list of nodes.

	// Strategy: Use a recursive function to serialize the node data manually, writing a type identifier first.

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
		Ar.ArIsSaveGame = false; // Fix: was true

		DeserializeNodes(Ar, LDTRoot, this);
	}
}

// Helper for serialization (needs to be added to header or kept private static)
// I'll add them as private member functions in .cpp or helper class?
// Since I can't easily change header now without another tool call, I'll use a static helper in .cpp
// But wait, I need access to private members of nodes? No, they are public properties or I can cast.
// Properties in nodes are public.

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

	// Serialize properties
	Node->Serialize(Ar);

	// For DecisionNode, we need to recursively serialize children
	if (NodeType == 2) // DecisionNode
	{
		ULearningDecisionTreeDecisionNode* DNode = Cast<ULearningDecisionTreeDecisionNode>(Node);
		SerializeNodes(Ar, DNode->Nodes);
	}

	// TableNode has NextNodes but they are transient during building?
	// In C#, TableNode `nextNodes` are used to build `lastNodes`.
	// But once built, `LDTRoot` contains the structure.
	// Wait, `TableNode` logic in C#:
	// `lastNodes[thisNodeIndex] = new DecisionNode(...)`
	// This replaces the TableNode in the parent list with a DecisionNode.
	// So a fully built tree should NOT contain TableNodes!
	// TableNodes are only temporary placeholders during construction.

	// If the tree is fully built, it consists of DecisionNodes and ActionNodes.
	// If `CreateDecisionTree` has finished, LDTRoot[0] should be a DecisionNode or ActionNode.
}

static ULearningDecisionTreeNode* DeserializeSingleNode(FArchive& Ar, UObject* Outer)
{
	uint8 NodeType = 0;
	Ar << NodeType;

	ULearningDecisionTreeNode* Node = nullptr;

	if (NodeType == 1) Node = NewObject<ULearningDecisionTreeTableNode>(Outer);
	else if (NodeType == 2) Node = NewObject<ULearningDecisionTreeDecisionNode>(Outer);
	else if (NodeType == 3) Node = NewObject<ULearningDecisionTreeActionNode>(Outer);

	if (Node)
	{
		Node->Serialize(Ar);

		if (NodeType == 2)
		{
			ULearningDecisionTreeDecisionNode* DNode = Cast<ULearningDecisionTreeDecisionNode>(Node);
			DeserializeNodes(Ar, DNode->Nodes, Outer);
		}
	}

	return Node;
}
