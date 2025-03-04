#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    INT,
    WHILE,
    IF,
    RETURN
} token;

typedef struct instruction {
    char var_name;
    token token_type;
    struct instruction* next;
} instruction;

typedef struct node {
    instruction* instructions;
    struct node** neighbors;
} node;

typedef struct {
    node* start;
} graph;

void block_analisys(instruction** instructions, FILE* fptr, char code[100]) {
    instruction* last = NULL;
    while (fgets(code, 100, fptr)) {
        if (code[0] == '}') break;

        instruction* new_instruction = (instruction*)malloc(sizeof(instruction));
        new_instruction->next = NULL;

        if (code[1] == 'i') { 
            new_instruction->var_name = code[5];
            new_instruction->token_type = INT;
        }

        if (*instructions == NULL) {
            *instructions = new_instruction;
        } else {
            last->next = new_instruction;
        }
        last = new_instruction;
    }
}

int main() {
    FILE* fptr;
    fptr = fopen("code.txt", "r");
    if (!fptr) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }

    char code[100];
    graph* structure = (graph*)malloc(sizeof(graph));
    structure->start = NULL;
    node* curr_node = structure->start;

    while (fgets(code, 100, fptr)) {
        node* new_node;
        if (code[0] == '{') {
            new_node = (node*)malloc(sizeof(node));
            new_node->instructions = NULL;
            new_node->neighbors = NULL;

            block_analisys(&new_node->instructions, fptr, code);
        }
        if (structure->start == NULL) {
            structure->start = new_node;
        } else {
            curr_node->neighbors = (node**)malloc(sizeof(node*));
            curr_node->neighbors[0] = new_node;
        }
        curr_node = new_node;
    }

    curr_node = structure->start;
    while(curr_node != NULL) {
        instruction* curr_instruction = curr_node->instructions;
        while (curr_instruction != NULL) {
            printf("var_name: %c, token_type: %d\n", curr_instruction->var_name, curr_instruction->token_type);
            curr_instruction = curr_instruction->next;
        }
        curr_node = curr_node->neighbors[0];
    }

    fclose(fptr);

    // curr_node = structure->start;
    // while (curr_node != NULL) {
    //     instruction* curr_instruction = curr_node->instructions;
    //     while (curr_instruction != NULL) {
    //         instruction* temp = curr_instruction;
    //         curr_instruction = curr_instruction->next;
    //         free(temp);
    //     }
    //     node* temp_node = curr_node;
    //     curr_node = curr_node->neighbors != NULL ? curr_node->neighbors[0] : NULL;
    //     free(temp_node);
    // }
    // free(structure);

    return 0;
}