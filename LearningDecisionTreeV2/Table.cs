using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class Table
{
    //Last 2 columns (Actions and dups)
    public Dictionary<string, List<int>> tableData = new Dictionary<string, List<int>>();
    private List<string> columnNames = new List<string>();
    private int totalRows = 0;
    public Table() { }

    public int GetTableRowCount
    {
        get
        {
            if (columnNames.Count > 0)
                return tableData[columnNames[0]].Count;
            return 0;
        }
    }
    public int GetTotalRowCount
    {
        get
        {
            return totalRows;
        }
    }
    public int GetStateCount(string column, int state)
    {
        if (tableData.ContainsKey(column))
        {
            int stateCount = 0;
            for (int row = 0; row < tableData[column].Count; row++)
                if (state == tableData[column][row])
                    stateCount += tableData[columnNames[tableData.Count - 1]][row];

            return stateCount;
        }

        return 0;
    }
    public int GetStateCount(int column, int state)
    {
        if (column < columnNames.Count - 1)
        {
            int stateCount = 0;
            for (int row = 0; row < tableData[columnNames[column]].Count; row++)
                if (state == tableData[columnNames[column]][row])
                    stateCount += tableData[columnNames[tableData.Count - 1]][row];

            return stateCount;
        }

        return 0;
    }
    public List<int> GetColumnStates(string column)
    {
        if (tableData.ContainsKey(column))
        {
            List<int> columnStates = new List<int>();

            foreach (int state in tableData[column])
                if (!columnStates.Contains(state))
                    columnStates.Add(state);

            return columnStates;
        }

        return null;
    }
    public List<int> GetColumnStates(int column)
    {
        if (column < columnNames.Count - 1)
        {
            List<int> columnStates = new List<int>();

            foreach (int state in tableData[columnNames[column]])
                if (!columnStates.Contains(state))
                    columnStates.Add(state);

            return columnStates;
        }

        return null;
    }
    public int GetNumberOfStates(string column)
    {
        if (tableData.ContainsKey(column))
        {
            List<int> columnStates = new List<int>();

            foreach (int state in tableData[column])
                if (!columnStates.Contains(state))
                    columnStates.Add(state);

            return columnStates.Count;
        }
        return 0;
    }
    public int GetNumberOfStates(int column)
    {
        if (column < columnNames.Count - 1)
        {
            List<int> columnStates = new List<int>();

            foreach (int state in tableData[columnNames[column]])
                if (!columnStates.Contains(state))
                    columnStates.Add(state);

            return columnStates.Count;
        }
        return 0;
    }

    public string GetColumnName(int column)
    {
        if (column < columnNames.Count - 1)
            return columnNames[column];

        Debug.Log("Invalid column index");
        return null;
    }

    public bool AddColumn(string name)
    {
        if (tableData.ContainsKey(name))
            return false;

        tableData.Add(name, new List<int>());
        columnNames.Add(name);
        return true;
    }
    public bool AddRow(params int[] row)
    {
        bool dup = false;
        int dupedData = 0;
        int dupedRow = 0;

        if (row.Length == tableData.Count - 1)
        {
            List<string> columnNames = new List<string>(tableData.Keys);
            for (int tableRow = 0; tableRow < tableData[columnNames[0]].Count; tableRow++)
            {
                for (int tableColumn = 0; tableColumn < columnNames.Count - 1; tableColumn++)
                {
                    if (tableData[columnNames[tableColumn]][tableRow] == row[tableColumn])
                        dupedData++;

                }

                if (dupedData == columnNames.Count - 1)
                {
                    dupedRow = tableRow;
                    dup = true;
                    break;
                }

                dupedData = 0;
            }

            if (dup)
                tableData[columnNames[columnNames.Count - 1]][dupedRow]++;
            else
            {
                for (int column = 0; column < row.Length; column++)
                    tableData[columnNames[column]].Add(row[column]);
                tableData[columnNames[columnNames.Count - 1]].Add(1);
            }

            totalRows++;
            return true;
        }
        return false;
    }

    public bool RemoveRow(int row)
    {
        //List<string> columnNames = new List<string>(table.Keys);
        if (row < tableData[columnNames[0]].Count)
        {
            totalRows -= tableData[columnNames[tableData.Count - 1]][row];
            for (int tableColumn = 0; tableColumn < columnNames.Count; tableColumn++)
                tableData[columnNames[tableColumn]].RemoveAt(row);
            return true;
        }
        return false;
    }
    public bool RemoveColumn(string column)
    {
        if (tableData.ContainsKey(column))
        {
            tableData.Remove(column);
            columnNames.Remove(column);
            RefreshTable();
            return true;
        }

        return false;
    }
    public bool RemoveColumn(int column)
    {
        //List<string> columnNames = new List<string>(table.Keys);
        if (tableData.Count > 0 && tableData.ContainsKey(columnNames[column]))
        {
            tableData.Remove(columnNames[column]);
            columnNames.RemoveAt(column);
            RefreshTable();
            return true;
        }
        return false;
    }

    public float IndividualStateProbability(string column, int state)
    {
        if (tableData.ContainsKey(column) && tableData[column].Contains(state))
        {
            int stateDups = 0;

            for (int row = 0; row < tableData[column].Count; row++)
                if (tableData[column][row] == state)
                    stateDups += tableData[columnNames[tableData.Count - 1]][row];

            return ((float)stateDups / totalRows);
        }

        return 0;
    }
    public float IndividualStateProbability(int column, int state)
    {
        if (column < columnNames.Count - 1 && tableData[columnNames[column]].Contains(state))
        {
            int stateDups = 0;

            for (int row = 0; row < tableData[columnNames[column]].Count; row++)
                if (tableData[columnNames[column]][row] == state)
                    stateDups += tableData[columnNames[tableData.Count - 1]][row];

            return ((float)stateDups / totalRows);
        }
        return 0;
    }

    public Table FilterTableByState(string column, int state)
    {
        Table newTable = this.DeepClone();

        if (tableData.ContainsKey(column) && tableData[column].Contains(state))
        {
            for (int row = 0; row < newTable.tableData[column].Count; row++)
                if (newTable.tableData[column][row] != state)
                {
                    newTable.RemoveRow(row);
                    row = -1;
                }
            return newTable;
        }
        Debug.Log("Error FilterTableByState");
        return null;
    }
    public Table FilterTableByState(int column, int state)
    {
        Table newTable = this.DeepClone();
        if (column < columnNames.Count - 1 && tableData[columnNames[column]].Contains(state))
        {
            for (int row = 0; row < newTable.tableData[columnNames[column]].Count; row++)
                if (newTable.tableData[columnNames[column]][row] != state)
                {
                    newTable.RemoveRow(row);
                    row = -1;
                }
            newTable.RemoveColumn(column);
            return newTable;
        }
        Debug.Log("Error FilterTableByState");
        return null;
    }



    public void RefreshTable()
    {
        //List<string> columnNames = new List<string>(table.Keys);
        int dupedStates = 0;
        if (tableData.Count > 1 && tableData[columnNames[0]].Count > 0)
            for (int selectedRow = 0; selectedRow < tableData[columnNames[0]].Count; selectedRow++)
                if (selectedRow < tableData[columnNames[0]].Count)
                    for (int row = selectedRow + 1; row < tableData[columnNames[0]].Count; row++)
                    {
                        for (int column = 0; column < columnNames.Count - 1; column++)
                        {
                            if (tableData[columnNames[column]][selectedRow] == tableData[columnNames[column]][row])
                            {
                                dupedStates++;
                            }
                        }

                        if (dupedStates == columnNames.Count - 1)
                        {
                            tableData[columnNames[columnNames.Count - 1]][selectedRow] += tableData[columnNames[columnNames.Count - 1]][row];
                            RemoveRow(row);
                            row = selectedRow;
                        }
                        dupedStates = 0;
                    }



    }

    public void DebugTable()
    {
        //List<string> columnNames = new List<string>(table.Keys);

        string debug = "";

        foreach (string columnName in columnNames)
            debug += columnName + " ";

        Debug.Log(debug);
        debug = "";

        if (tableData.Count > 0)
            for (int tableRow = 0; tableRow < tableData[columnNames[0]].Count; tableRow++)
            {

                for (int tableColumn = 0; tableColumn < columnNames.Count; tableColumn++)
                    debug += columnNames[tableColumn] + " : " + tableData[columnNames[tableColumn]][tableRow] + "|";

                Debug.Log(debug);
                debug = "";
            }
    }

}
