using System;
using System.Collections.Generic;

public class Reproduction
{
    public static void Main(string[] args)
    {
        Table table = new Table();
        // Add 2 data columns + Duplicates column (added by default? No, explicit?)
        // Let's check AddColumn
        // AddColumn adds to tableData and columnNames.

        // LDTV2Manager adds "Duplicates" explicitly.
        table.AddColumn("Col1");
        table.AddColumn("Col2");
        table.AddColumn("Duplicates");

        Console.WriteLine("Columns: " + table.tableData.Keys.Count);

        // AddRow expects row.Length == tableData.Count - 1 (excluding Duplicates)
        table.AddRow(1, 1);
        table.AddRow(2, 2);
        table.AddRow(3, 3);

        Console.WriteLine("Rows before removal: " + table.GetTableRowCount);

        // Print internal state before
        PrintTableState(table, "Before Removal");

        // Remove first row
        Console.WriteLine("Removing row 0...");
        bool result = table.RemoveRow(0);
        Console.WriteLine("RemoveRow returned: " + result);

        Console.WriteLine("Rows after removal: " + table.GetTableRowCount);

        // Print internal state after
        PrintTableState(table, "After Removal");

        // Verification
        var keys = new List<string>(table.tableData.Keys);
        bool consistent = true;
        int expectedCount = table.tableData[keys[0]].Count;

        foreach(var key in keys)
        {
            if (table.tableData[key].Count != expectedCount)
            {
                consistent = false;
                Console.WriteLine($"Mismatch in column {key}: has {table.tableData[key].Count}, expected {expectedCount}");
            }
        }

        if (!consistent)
        {
            Console.WriteLine("BUG DETECTED: Columns have different lengths.");
            Environment.Exit(1);
        }
        else
        {
            Console.WriteLine("Table is consistent.");
        }
    }

    static void PrintTableState(Table table, string label)
    {
        Console.WriteLine($"--- {label} ---");
        foreach(var kvp in table.tableData)
        {
            Console.Write($"{kvp.Key}: ");
            foreach(var val in kvp.Value)
            {
                Console.Write(val + " ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("----------------");
    }
}
