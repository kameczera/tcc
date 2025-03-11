# Liveness Analysis Program

This program analyzes the **liveness** of variables in a program written in a language similar to C. The analysis determines which variables are **alive** (i.e., still needed) at different points in the program.

## Features
- Parses a simple C-like program.
- Performs liveness analysis on variables.
- Outputs the `in` and `out` sets for each block in the control flow graph.

## Input Format
The input consists of:
- Variable declarations inside `{}`.
- Labels (`LABEL`) defining basic blocks.
- Conditional jumps (`COND`) for flow control.
- A `RETURN` statement.

### Example Input:
```
{
    int z = 10;
    int x = 0;
    int y = 1;
}
LABEL L0
{
    z = x * 2 + y;
    x++;
    y = x + z;
}
COND a < N L0 L1
LABEL L1
RETURN c
```

## Output Format
The program outputs the liveness analysis results for each node in the control flow graph (CFG), showing the `in` and `out` sets of variables.

### Example Output:
```
NODE BLOCK:
in: n
out: x,y,n

NODE COND:
in: x, y, n
out: x, y, n

NODE BLOCK:
in: x, y, n
out: x, z, n

NODE RETURN:
in: x, z, n
out: x, y, n
```

## Compilation
To compile the program, run:
```
gcc liveness.c -o liveness
```

## Execution

To run the program, use the following command:

```
./liveness -f <code_file> -d1 -d2 -d3
```

## Parameters:
- -f <code_file>: Specifies the code file to be analyzed (e.g., code.txt).
- -d1: Enables the display of the constructed graph.
- -d2: Enables the display of the kill and gen tables.
- -d3: Enables the display of the final IN and OUT tables.

## Example:

```
./liveness -f code.txt -d1 -d2 -d3
```

## This command runs the program using code.txt as input and enables all display options.

## Requirements
- **GCC Compiler** (for compilation)
- **C Standard Library**

## ✅ TODO List
- ✅ **Parser**
- ✅ **Graph Construction**
- 🟡 **Algorithm Implementation**
- ❌ **Refactoring**

## License
This project is open-source and free to use.
