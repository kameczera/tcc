# Kaleidoscope - LLVM Project in C++

This project is a simple implementation of a compiler and interpreter for a custom programming language, following the official LLVM tutorial "My First Language Frontend."

## Overview

Kaleidoscope is a minimal programming language designed to demonstrate the core concepts of the compilation process. The project uses LLVM to build a frontend that can parse, analyze, and generate machine code from a basic mathematical expression language.

## Prerequisites

Before getting started, ensure you have the following installed on your machine:

- **LLVM**: The project requires LLVM libraries for building and compiling the frontend. Please install LLVM version 11 or higher.
- **C++ Compiler**: You need a C++ compiler like `g++` or `clang++` to compile the project.
- **CMake**: This is required to build the project using a simple and efficient build process.

## Installation

### 1. Build the project:

This will compile the source code and generate the necessary executables.
Usage

Once the project is built, you can run the Kaleidoscope interpreter and compiler from the command line:
Running the Interpreter

To run the interpreter, use the following command:

```
./kaleidoscope
```

You can now enter expressions like:

```
1 + 2 * 3
```