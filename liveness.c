#include <stdio.h>

typedef enum {
    INT,
    WHILE,
    IF,
    RETURN
} token;

typedef struct {
    char var_name;
    token token_type;
    struct instruction* next;
} instruction;

typedef struct {
    instruction* instructions;
    struct node** neighbors;
} node;

typedef struct {
    node* start;
} graph;

void block_analisys() {

}

int main() {
    FILE* fptr;
    fptr = fopen("code.txt", "r");
    char code[100];
    graph* structure = (graph*) malloc(sizeof(graph));
    structure->start = (node*) malloc(sizeof(node));
    node* curr_node = structure->start;
    curr_node->instructions = (instruction*) malloc(sizeof(instruction));
    while(fgets(code, 100, fptr)) {
        if(code[0] == '{') {
            instruction* last_instruction;
            fgets(code, 100, fptr);
            while(code[0] != '}') {
                if(code[1] == 'i') {
                    curr_node->instructions->var_name = code[5];
                    last_instruction = curr_node->instructions;
                }
                fgets(code, 100, fptr);
            }
        }
        printf("%s", code);
    }


    fclose(fptr);
    return 0;
}