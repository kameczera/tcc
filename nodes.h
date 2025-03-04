#ifndef NODES_H
#define NODES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NODE_BLOCK,
    NODE_COND,
    NODE_LABEL
} node_type;

typedef struct instruction {
    char var_name;
    struct instruction* next;
} instruction;

typedef struct {
    instruction* instructions;
} block_node_data;

typedef struct {
    int labels[2];
} cond_node_data;

typedef struct {
    int label_id;
} label_node_data;

typedef struct node {
    node_type type;
    struct node** neighbors;
    int neighbor_count;
    union {
        block_node_data block_data;
        cond_node_data cond_data;
        label_node_data label_data;
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
    new_node->cond_data.labels[0] = NULL;
    new_node->cond_data.labels[1] = NULL;
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

void print_node(const node* n) {
    switch (n->type) {
        case NODE_BLOCK:
            printf("Node Type: BLOCK\n");
            instruction* instr = n->block_data.instructions;
            while (instr != NULL) {
                printf("  var_name: %c\n", instr->var_name);
                instr = instr->next;
            }
            break;
        case NODE_COND:
            printf("Node Type: COND\n");
            printf("  condition LABEL 1: %i\n", n->cond_data.labels[0]);
            printf("  condition LABEL 2: %i\n", n->cond_data.labels[1]);
            break;
        case NODE_LABEL:
            printf("Node Type: LABEL\n");
            printf("  label_id: %d\n", n->label_data.label_id);
            break;
        default:
            printf("Node Type: Unknown\n");
            break;
    }
}


#endif