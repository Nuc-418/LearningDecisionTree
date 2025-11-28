#include "Misc/AutomationTest.h"
#include "LearningDecisionTree.h"
#include "LearningDecisionTreeTable.h"
#include "Async/LearningDecisionTreeTrainer.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLearningDecisionTreeAsyncTest, "LearningDecisionTree.AsyncTest", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLearningDecisionTreeAsyncTest::RunTest(const FString& Parameters)
{
	// 1. Setup a Table with known data (XOR-like pattern)
	FLearningDecisionTreeTable TestTable;
	TestTable.AddColumn(FName("A"));
	TestTable.AddColumn(FName("B"));
	TestTable.AddColumn(FName("Action")); // Last column is Action

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

	// 3. Verify Structure
	// Root should be a DecisionNode because Entropy > 0
	if (ShadowRoot->GetType() != FShadowNode::DecisionNode)
	{
		AddError(TEXT("Root should be a DecisionNode for XOR data."));
		return false;
	}

	// 4. Manual Verification of ID3 Logic
	// If A is split first:
	//   A=0: (0,0)->0, (0,1)->1 => Split on B
	//   A=1: (1,0)->1, (1,1)->0 => Split on B

	TSharedPtr<FShadowDecisionNode, ESPMode::ThreadSafe> DNode = StaticCastSharedPtr<FShadowDecisionNode>(ShadowRoot);
	// We don't strictly care if it picked A(0) or B(1) as best column, both have equal gain.
	// But it must have children.
	if (DNode->NextNodes.Num() != 2)
	{
		AddError(TEXT("Root DecisionNode should have 2 children (0 and 1)."));
		return false;
	}

    // 5. Verify Inconsistent Data Handling (Infinite Loop Fix)
    FLearningDecisionTreeTable InconsistentTable;
    InconsistentTable.AddColumn(FName("X"));
    InconsistentTable.AddColumn(FName("Action"));
    // Same state, different actions
    InconsistentTable.AddRow({0, 0});
    InconsistentTable.AddRow({0, 1});

    TSharedPtr<FShadowNode, ESPMode::ThreadSafe> InconsistentRoot = FLearningDecisionTreeTrainer::Train(InconsistentTable);
    if (!InconsistentRoot.IsValid())
    {
        AddError(TEXT("Inconsistent training returned null."));
        return false;
    }
    // Should be an ActionNode (Leaf) because we cannot split further
    if (InconsistentRoot->GetType() != FShadowNode::ActionNode)
    {
        AddError(TEXT("Inconsistent data should result in an ActionNode (Leaf), not infinite splitting."));
        return false;
    }

	return true;
}
