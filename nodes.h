#ifndef NODES_H
#define NODES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NODE_BLOCK,
    NODE_COND,
    NODE_LABEL,
    NODE_RETURN,
    NODE_INT
} node_type;

typedef struct instruction {
    char var_name;
    char* operands;
    struct instruction* next;
} instruction;

typedef struct {
    instruction* instructions;
} block_node_data;

typedef struct {
    int labels[2];
    char* operands;
} cond_node_data;

typedef struct {
    int label_id;
} label_node_data;

typedef struct {
    char var_name;
} return_node_data;

typedef struct {
    instruction* instruction;
} int_node_data;

typedef struct node {
    node_type type;
    struct node** neighbors;
    int neighbor_count;
    union {
        block_node_data block_data;
        int_node_data int_data;
        cond_node_data cond_data;
        label_node_data label_data;
        return_node_data return_data;
    };
} node;

node* create_block_node() {
    node* new_node = (node*)malloc(sizeof(node));
    if (!new_node) {
        perror("Erro ao alocar memória para o nó");
        exit(1);
    }
    new_node->type = NODE_BLOCK;
    new_node->neighbors = NULL;
    new_node->neighbor_count = 0;
    new_node->block_data.instructions = NULL;
    return new_node;
}

node* create_cond_node() {
    node* new_node = (node*)malloc(sizeof(node));
    if (!new_node) {
        perror("Erro ao alocar memória para o nó");
        exit(1);
    }
    new_node->type = NODE_COND;
    new_node->neighbors = NULL;
    new_node->neighbor_count = 0;
    new_node->cond_data.labels[0] = 0;
    new_node->cond_data.labels[1] = 0;
    return new_node;
}

node* create_label_node() {
    node* new_node = (node*)malloc(sizeof(node));
    if (!new_node) {
        perror("Erro ao alocar memória para o nó");
        exit(1);
    }
    new_node->type = NODE_LABEL;
    new_node->neighbors = NULL;
    new_node->neighbor_count = 0;
    return new_node;
}

node* create_return_data() {
    node* new_node = (node*)malloc(sizeof(node));
    if (!new_node) {
        perror("Erro ao alocar memória para o nó");
        exit(1);
    }
    new_node->type = NODE_RETURN;
    new_node->neighbors = NULL;
    new_node->neighbor_count = 0;
    return new_node;
}

node* create_int_data() {
    node* new_node = (node*)malloc(sizeof(node));
    if (!new_node) {
        perror("Erro ao alocar memória para o nó");
        exit(1);
    }
    new_node->type = NODE_INT;
    new_node->neighbors = NULL;
    new_node->neighbor_count = 0;
    new_node->int_data.instruction = NULL;
    return new_node;
}

void print_node(const node* n) {
    switch (n->type) {
        case NODE_BLOCK:
            printf("Node Type: BLOCK\n");
            instruction* instr = n->block_data.instructions;
            while (instr != NULL) {
                printf("    var_name: %c, ", instr->var_name);
                printf("operands: %c %c\n", instr->operands[0], instr->operands[1]);
                instr = instr->next;
            }
            break;
        case NODE_COND:
            printf("Node Type: COND\n");
            printf("    condition LABEL 1: %i\n", n->cond_data.labels[0]);
            printf("    condition LABEL 2: %i\n", n->cond_data.labels[1]);
            printf("    operands: %c, %c", n->cond_data.operands[0], n->cond_data.operands[1]);
            break;
        case NODE_LABEL:
            printf("Node Type: LABEL\n");
            printf("    label_id: %d\n", n->label_data.label_id);
            break;
        case NODE_RETURN:
            printf("Node Type: RETURN\n");
            printf("    var_name: %c\n", n->return_data.var_name);
            break;
        case NODE_INT:
            printf("Node Type: INT\n");
            printf("    var_name: %c, ", n->int_data.instruction->var_name);
            printf("operands: %c %c\n", n->int_data.instruction->operands[0], n->int_data.instruction->operands[1]);
            break;
        default:
            printf("Node Type: Unknown\n");
            break;
    }
}


#endif