Liveness Analysis for C-like Language
This project implements a liveness analysis tool for variables in a program written in a language similar to C. It analyzes the liveness of variables at different points in the program, helping to track which variables are "alive" (used or modified) at different locations (like at the entry and exit points of blocks or conditions).

The tool processes a program, identifies the variables' liveness in each block and condition, and provides the results in a structured output.

Features
Analyzes the liveness of variables at different program locations (blocks and conditions).
Outputs variable states at entry and exit points of each block and condition.
Provides a simplified approach for understanding how variables are used or modified during the execution flow of the program.
Example Input
c
Copiar
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
Example Output
yaml
Copiar
NODE BLOCK:
in: n
out: x, y, n
NODE COND:
in: x, y, n
out: x, y, n
NODE BLOCK:
in: x, y, n
out: x, z, n
NODE RETURN:
in: x, z, n
out: x, y, n
Compilation
To compile the program, use the following command:

bash
Copiar
gcc liveness.c -o liveness
This will generate an executable file named liveness.

Execution
Once compiled, you can run the tool with:

bash
Copiar
.\liveness.exe
<<<<<<< HEAD
This will execute the program and print the liveness analysis results based on the input program.
=======
This will execute the program and print the liveness analysis results based on the input program.
>>>>>>> af63b86715027b3da2193e2733073aefc7b67367
