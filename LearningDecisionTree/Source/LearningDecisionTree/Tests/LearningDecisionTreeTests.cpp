#include "Misc/AutomationTest.h"
#include "LearningDecisionTree.h"
#include "LearningDecisionTreeTable.h"
#include "Async/LearningDecisionTreeTrainer.h"

/**
 * Test the async shadow tree training with XOR-like data pattern.
 * Verifies that the ID3 algorithm correctly builds a decision tree.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLearningDecisionTreeAsyncTest, "LearningDecisionTree.AsyncTraining", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLearningDecisionTreeAsyncTest::RunTest(const FString& Parameters)
{
	// 1. Setup a Table with known data (XOR-like pattern)
	// New API: No Duplicates column needed - it's managed internally
	FLearningDecisionTreeTable TestTable;
	TestTable.AddColumn(FName("A"));
	TestTable.AddColumn(FName("B"));
	TestTable.AddColumn(FName("Action")); // Last column is always Action

	// A=0, B=0 -> Action=0
	TestTable.AddRow({0, 0, 0});
	// A=0, B=1 -> Action=1
	TestTable.AddRow({0, 1, 1});
	// A=1, B=0 -> Action=1
	TestTable.AddRow({1, 0, 1});
	// A=1, B=1 -> Action=0
	TestTable.AddRow({1, 1, 0});

	// 2. Run Shadow Training directly (Synchronously for test purpose)
	TSharedPtr<FShadowNode, ESPMode::ThreadSafe> ShadowRoot = FLearningDecisionTreeTrainer::Train(TestTable);

	if (!ShadowRoot.IsValid())
	{
		AddError(TEXT("Shadow Training returned null root."));
		return false;
	}

	// 3. Verify Structure - Root should be a DecisionNode because Entropy > 0
	if (ShadowRoot->GetType() != FShadowNode::DecisionNode)
	{
		AddError(TEXT("Root should be a DecisionNode for XOR data."));
		return false;
	}

	// 4. Verify DecisionNode has correct children
	TSharedPtr<FShadowDecisionNode, ESPMode::ThreadSafe> DNode = StaticCastSharedPtr<FShadowDecisionNode>(ShadowRoot);
	if (DNode->NextNodes.Num() != 2)
	{
		AddError(TEXT("Root DecisionNode should have 2 children (states 0 and 1)."));
		return false;
	}

	return true;
}

/**
 * Test that inconsistent data (same features, different actions) doesn't cause infinite loops.
 * The algorithm should create an ActionNode with probabilistic output.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLearningDecisionTreeInconsistentDataTest, "LearningDecisionTree.InconsistentData", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLearningDecisionTreeInconsistentDataTest::RunTest(const FString& Parameters)
{
	FLearningDecisionTreeTable InconsistentTable;
	InconsistentTable.AddColumn(FName("X"));
	InconsistentTable.AddColumn(FName("Action"));

	// Same feature state (X=0), but different actions - inconsistent data
	InconsistentTable.AddRow({0, 0});
	InconsistentTable.AddRow({0, 1});

	TSharedPtr<FShadowNode, ESPMode::ThreadSafe> InconsistentRoot = FLearningDecisionTreeTrainer::Train(InconsistentTable);

	if (!InconsistentRoot.IsValid())
	{
		AddError(TEXT("Inconsistent training returned null."));
		return false;
	}

	// Should be an ActionNode (Leaf) because we cannot split further with 0 info gain
	if (InconsistentRoot->GetType() != FShadowNode::ActionNode)
	{
		AddError(TEXT("Inconsistent data should result in an ActionNode (Leaf), not infinite splitting."));
		return false;
	}

	// Verify the ActionNode has both actions with their counts
	TSharedPtr<FShadowActionNode, ESPMode::ThreadSafe> ANode = StaticCastSharedPtr<FShadowActionNode>(InconsistentRoot);
	if (ANode->ActionNames.Num() != 2)
	{
		AddError(TEXT("ActionNode should have 2 possible actions for inconsistent data."));
		return false;
	}

	return true;
}

/**
 * Test duplicate row handling - same row added multiple times should increase weight.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLearningDecisionTreeDuplicateTest, "LearningDecisionTree.DuplicateHandling", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLearningDecisionTreeDuplicateTest::RunTest(const FString& Parameters)
{
	FLearningDecisionTreeTable Table;
	Table.AddColumn(FName("Feature"));
	Table.AddColumn(FName("Action"));

	// Add same row 3 times
	Table.AddRow({1, 5});
	Table.AddRow({1, 5});
	Table.AddRow({1, 5});

	// Should have 1 physical row but TotalRows = 3
	if (Table.GetTableRowCount() != 1)
	{
		AddError(FString::Printf(TEXT("Expected 1 physical row, got %d"), Table.GetTableRowCount()));
		return false;
	}

	if (Table.GetTotalRowCount() != 3)
	{
		AddError(FString::Printf(TEXT("Expected TotalRows=3, got %d"), Table.GetTotalRowCount()));
		return false;
	}

	if (Table.GetDuplicateCount(0) != 3)
	{
		AddError(FString::Printf(TEXT("Expected DuplicateCount=3, got %d"), Table.GetDuplicateCount(0)));
		return false;
	}

	return true;
}

/**
 * Test basic table operations - AddColumn, AddRow, GetColumnStates.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLearningDecisionTreeTableTest, "LearningDecisionTree.TableOperations", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLearningDecisionTreeTableTest::RunTest(const FString& Parameters)
{
	FLearningDecisionTreeTable Table;

	// Test AddColumn
	TestTrue(TEXT("AddColumn should succeed"), Table.AddColumn(FName("Col1")));
	TestTrue(TEXT("AddColumn should succeed"), Table.AddColumn(FName("Col2")));
	TestFalse(TEXT("Duplicate column should fail"), Table.AddColumn(FName("Col1")));

	// Test AddRow
	TestTrue(TEXT("AddRow should succeed with correct size"), Table.AddRow({1, 2}));
	TestFalse(TEXT("AddRow should fail with wrong size"), Table.AddRow({1}));
	TestFalse(TEXT("AddRow should fail with wrong size"), Table.AddRow({1, 2, 3}));

	// Test GetColumnStates
	Table.AddRow({1, 3});
	Table.AddRow({2, 2});

	TArray<int32> Col1States = Table.GetColumnStates(FName("Col1"));
	TestEqual(TEXT("Col1 should have 3 unique states"), Col1States.Num(), 3);

	return true;
}
