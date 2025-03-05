#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nodes.h"

void block_analisys(instruction** instructions, FILE* fptr, char code[100]) {
    instruction* last = NULL;
    while (fgets(code, 100, fptr)) {
        if (code[0] == '}') break;

        instruction* new_instruction = (instruction*)malloc(sizeof(instruction));
        new_instruction->next = NULL;

        if (code[1] == 'i') { 
            new_instruction->var_name = code[5];
            int cont_str = 9;
            int cont_operands = 0;
            new_instruction->operands = (char*)malloc(sizeof(char) * 2);
            new_instruction->operands[0] = '\0';
            new_instruction->operands[1] = '\0';
            while(1) {
                if(code[cont_str] > 96 && code[cont_str] < 123 && cont_operands < 2) {
                    new_instruction->operands[cont_operands] = code[cont_str];
                    cont_operands++;
                }
                else if(code[cont_str] == '\0') break;
                cont_str++;
            }
            new_instruction->next = NULL;
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

    // PARSE

    char code[100];
    node** list_of_nodes = (node*)malloc(sizeof(node) * 10);
    int cont = 0;

    while (fgets(code, 100, fptr)) {
        node* new_node;
        if (code[0] == '{') {
            new_node = create_block_node();
            block_analisys(&new_node->block_data.instructions, fptr, code);
        } else if (code[0] == 'c') {
            new_node = create_cond_node();
            new_node->cond_data.labels[0] = code[12] - '0';
            new_node->cond_data.labels[1] = code[15] - '0';
        } else if (code[0] == 'L') {
            int label = code[1] - '0';
            new_node = create_label_node();
            new_node->label_data.label_id = label;
        } else if (code[0] == 'R') {
            new_node = create_return_data();
            new_node->return_data.var_name = code[7];
        }

        if (new_node != NULL) {
            list_of_nodes[cont] = new_node;
            cont++;
        }
    }

    fclose(fptr);

    // CONSTRUÇÃO DO GRAFO
    
    node* start = list_of_nodes[0];
    node* curr_node;
    for(int i = 0; i < cont - 1; i++) {
        curr_node = list_of_nodes[i];
        if(curr_node->type == NODE_COND) {
            curr_node->neighbors = (node**)malloc(sizeof(node*) * 2);
            node* ptr_label;
            int cont_neighbors = 0;
            for(int j = 0; j < cont; j++) {
                ptr_label = list_of_nodes[j];
                if(ptr_label->type == NODE_LABEL && (curr_node->cond_data.labels[0] == ptr_label->label_data.label_id || curr_node->cond_data.labels[1] == ptr_label->label_data.label_id)) {
                    curr_node->neighbors[cont_neighbors] = ptr_label;
                    cont_neighbors++;
                }
            }
        } else {
            curr_node->neighbors = (node**)malloc(sizeof(node*));
            curr_node->neighbors[0] = list_of_nodes[i + 1];
        }
    }
    curr_node = start;
    for (int i = 0; i < cont; i++) {
        printf("Nó atual:\n");
        print_node(list_of_nodes[i]);
        if (list_of_nodes[i]->type == NODE_COND) {
            printf("Neighbors:\n");
            print_node(list_of_nodes[i]->neighbors[0]);
            print_node(list_of_nodes[i]->neighbors[1]);
        } else if (list_of_nodes[i]->neighbors != NULL) {
            printf("Neighbor:\n");
            print_node(list_of_nodes[i]->neighbors[0]);
        }
    }
    
    free(list_of_nodes);
    
    // ALGORITMO



    return 0;
}