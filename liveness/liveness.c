#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nodes.h"

void get_operands(char** operands, int *cont_operands, char code[100], int cont_str) {
    int tmp = cont_str;
    // passando contando a quantidade de variáveis
    while(1) {
        if(code[cont_str] >= 'a' && code[cont_str] <= 'z') {
            (*cont_operands)++;
        }
        else if(code[cont_str] == '\0') break;
        cont_str++;
    }

    *operands = (char*)malloc(sizeof(char) * (*cont_operands));
    // passando novamente salvando os caracteres
    cont_str = tmp;
    int current_operand = 0;
    while(1) {
        if(code[cont_str] >= 'a' && code[cont_str] <= 'z') {
            (*operands)[current_operand] = code[cont_str];

            current_operand++;
        }
        else if(code[cont_str] == '\0') break;
        cont_str++;
    }
}

void get_instruction(instruction* new_instruction, char code[100]) {
    int cont_str = 0;
    while(1) {
        if(code[cont_str] == 'i') {
            new_instruction->var_name = code[cont_str + 4];
            cont_str += 5;
            break;
        }
        cont_str++;
    }
    get_operands(&new_instruction->operands, &new_instruction->cont_operands, code, cont_str);
    new_instruction->next = NULL;
}

void block_analisys(instruction** instructions, FILE* fptr, char code[100]) {
    instruction* last = NULL;
    while (fgets(code, 100, fptr)) {
        if (code[0] == '}') break;

        instruction* new_instruction = (instruction*)malloc(sizeof(instruction));
        new_instruction->next = NULL;
        new_instruction->cont_operands = 0;

        if (code[1] == 'i') { 
            get_instruction(new_instruction, code);
        }

        if (*instructions == NULL) {
            *instructions = new_instruction;
        } else {
            last->next = new_instruction;
        }
        last = new_instruction;
    }
}

int equal_ptrs(int** ptr, int** tmp_ptr, int node_cont, int vars) {
    for(int i = 0; i < node_cont; i++) {
        for(int j = 0; j < vars; j++) {
            if(ptr[i][j] != tmp_ptr[i][j]) return 0;
        }
    }
    return 1;
}

void copy_ptrs(int** dest, int** src, int node_cont, int vars) {
    for (int i = 0; i < node_cont; i++) {
        for (int j = 0; j < vars; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void clean(int** dest, int node_cont, int vars) {
    for (int i = 0; i < node_cont; i++) {
        for (int j = 0; j < vars; j++) {
            dest[i][j] = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    int debug_graph = 0, debug_kill_gen = 0, debug_algorithm = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d1") == 0) debug_graph = 1;
        if (strcmp(argv[i], "-d2") == 0) debug_kill_gen = 1;
        if (strcmp(argv[i], "-d3") == 0) debug_algorithm = 1;
    }

    FILE* fptr = fopen("code.txt", "r");
    if (!fptr) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }

    // LEXER
    char code[100];
    node** list_of_nodes = (node**)malloc(sizeof(node) * 10); // TODO: tamanho dinamico
    int node_cont = 0;

    while (fgets(code, 100, fptr)) {
        node* new_node;
        if (code[0] == '{') {
            new_node = create_block_node();
            block_analisys(&new_node->block_data.instructions, fptr, code);
        } else if (code[0] == 'C') {
            new_node = create_cond_node();
            new_node->cond_data.cont_operands = 0;
            get_operands(&new_node->cond_data.operands, &new_node->cond_data.cont_operands, code, 4);
            new_node->cond_data.labels[0] = code[12] - '0';
            new_node->cond_data.labels[1] = code[15] - '0';
        } else if (code[0] == 'L') {
            int label = code[1] - '0';
            new_node = create_label_node();
            new_node->label_data.label_id = label;
        } else if (code[0] == 'R') {
            new_node = create_return_data();
            new_node->return_data.var_name = code[7];
        } else if(code[0] == 'i') {
            new_node = create_int_data();
            new_node->int_data.instruction = (instruction*)malloc(sizeof(instruction));
            new_node->int_data.instruction->cont_operands = 0;
            get_instruction(new_node->int_data.instruction, code);
        }

        if (new_node != NULL) {
            list_of_nodes[node_cont] = new_node;
            node_cont++;
        }
    }

    fclose(fptr);

    // PARSER

    node* start = list_of_nodes[0];
    node* curr_node;
    int** succ = (int*)malloc(sizeof(int*) * node_cont);
    
    for(int i = 0; i < node_cont - 1; i++) {
        curr_node = list_of_nodes[i];
        if(curr_node->type == NODE_COND) {
            succ[i] = (int*) malloc(sizeof(int) * 2);
            curr_node->neighbors = (node**)malloc(sizeof(node*) * 2);
            node* ptr_label;
            int cont_neighbors = 0;
            for(int j = 0; j < node_cont; j++) {
                ptr_label = list_of_nodes[j];
                if(ptr_label->type == NODE_LABEL && (curr_node->cond_data.labels[0] == ptr_label->label_data.label_id || curr_node->cond_data.labels[1] == ptr_label->label_data.label_id)) {
                    curr_node->neighbors[cont_neighbors] = ptr_label;
                    succ[i][cont_neighbors] = j;
                    cont_neighbors++;
                }
            }
        } else {
            succ[i] = (int*) malloc(sizeof(int));
            curr_node->neighbors = (node**)malloc(sizeof(node*));
            succ[i][0] = i + 1;
            curr_node->neighbors[0] = list_of_nodes[i + 1];
        }
    }
    curr_node = start;

    // ----------------------------- debug graph building ----------------------------- //
    if (debug_graph) {
        for (int i = 0; i < node_cont; i++) {
            printf("%i. No atual:\n", i + 1);
            print_node(list_of_nodes[i]);
            printf("\n");
            if (list_of_nodes[i]->type == NODE_COND) {
                printf("Neighbors:\n");
                printf(" %i. ", succ[i][0] + 1);
                print_node(list_of_nodes[i]->neighbors[0]);
                printf(" %i. ", succ[i][1] + 1);
                print_node(list_of_nodes[i]->neighbors[1]);
                printf("\n");
            } else if (list_of_nodes[i]->neighbors != NULL) {
                printf("Neighbor:\n");
                printf(" %i. ", succ[i][0] + 1);
                print_node(list_of_nodes[i]->neighbors[0]);
                printf("\n");
            }
        }
    }
    // -------------------------------------------------------------------------------- //

    // ALGORITMO

    char** gen = (char**)malloc(sizeof(char*) * node_cont);
    for(int i = 0; i < node_cont; i++) {
        if(list_of_nodes[i]->type == NODE_COND) gen[i] = (char*)malloc(sizeof(char) * list_of_nodes[i]->cond_data.cont_operands);
        else if(list_of_nodes[i]->type == NODE_INT) gen[i] = (char*)malloc(sizeof(char) * list_of_nodes[i]->int_data.instruction->cont_operands);
        else if(list_of_nodes[i]->type == NODE_RETURN) gen[i] = (char*)malloc(sizeof(char) * 2);
        else gen[i] = NULL;
    }
    char* kill = (char*)malloc(sizeof(char) * node_cont);
    
    for(int i = 0; i < node_cont; i++) {
        switch (list_of_nodes[i]->type) {
            case NODE_RETURN:
                kill[i] = '\0';
                gen[i][0] = list_of_nodes[i]->return_data.var_name;
                gen[i][1] = '\0';
                break;
            case NODE_INT:
                kill[i] = list_of_nodes[i]->int_data.instruction->var_name;
                for(int j = 0; j < list_of_nodes[i]->int_data.instruction->cont_operands; j++) {
                    gen[i][j] = list_of_nodes[i]->int_data.instruction->operands[j];
                }
                break;
            case NODE_COND:
                kill[i] = '\0';
                for(int j = 0; j < list_of_nodes[i]->cond_data.cont_operands; j++) {
                    gen[i][j] = list_of_nodes[i]->cond_data.operands[j];
                }
                break;
            default:
                kill[i] = '\0';
                break;
        }
    }

    // ----------------------------- debug kill & gen table ----------------------------- //
    if (debug_kill_gen) {
        for(int i = 0; i < node_cont; i++) {
            if(list_of_nodes[i]->type == NODE_COND) {
                for(int j = 0; j < list_of_nodes[i]->cond_data.cont_operands; j++) {
                    printf("gen[%i][%i] = %c, ", i + 1, j, gen[i][j]);
                }
                printf("\n");
                printf("kill[%i] = %c\n", i + 1, kill[i]);
            }
            else if(list_of_nodes[i]->type == NODE_INT){
                for(int j = 0; j < list_of_nodes[i]->int_data.instruction->cont_operands; j++) {
                    printf("gen[%i][%i] = %c, ", i + 1, j, gen[i][j]);
                }
                printf("\n");
                printf("kill[%i] = %c\n", i + 1, kill[i]);
            }
            else if(list_of_nodes[i]->type == NODE_RETURN){
                printf("gen[%i] = %c\n", i + 1, gen[i][0]);
            }
        }
    }
    // ---------------------------------------------------------------------------------- //

    int cont_variables = 0;
    int hash_alphabet[26];
    for(int i = 0; i < 26; i++) hash_alphabet[i] = -1;
    for(int i = 0; i < node_cont; i++) {
        if(list_of_nodes[i]->type == NODE_INT) {
            if(hash_alphabet[list_of_nodes[i]->int_data.instruction->var_name - 'a'] == -1) {
                hash_alphabet[list_of_nodes[i]->int_data.instruction->var_name - 'a'] = cont_variables;
                cont_variables++;
            }
        }
    }

    int** in = (int**)malloc(sizeof(int*) * node_cont);
    for(int i = 0; i < node_cont; i++) {
        in[i] = (int*)malloc(sizeof(int) * cont_variables);
        for(int j = 0; j < cont_variables; j++) {
            in[i][j] = 0;
        }
    }
    int** out = (int**)malloc(sizeof(int) * node_cont);
    for(int i = 0; i < node_cont; i++) {
        out[i] = (int*)malloc(sizeof(int) * cont_variables);
        for(int j = 0; j < cont_variables; j++) {
            out[i][j] = 0;
        }
    }

    int** tmp_in = (int**)malloc(sizeof(int*) * node_cont);
    for(int i = 0; i < node_cont; i++) {
        tmp_in[i] = (int*)malloc(sizeof(int) * cont_variables);
        for(int j = 0; j < cont_variables; j++) {
            tmp_in[i][j] = 0;
        }
    }
    int** tmp_out = (int**)malloc(sizeof(int) * node_cont);
    for(int i = 0; i < node_cont; i++) {
        tmp_out[i] = (int*)malloc(sizeof(int) * cont_variables);
        for(int j = 0; j < cont_variables; j++) {
            tmp_out[i][j] = 0;
        }
    }
    do {
        copy_ptrs(tmp_in, in, node_cont, cont_variables);
        copy_ptrs(tmp_out, out, node_cont, cont_variables);
        clean(in, node_cont, cont_variables);
        clean(out, node_cont, cont_variables);
        in[node_cont - 1][hash_alphabet[*gen[node_cont - 1] - 'a']] = 1;
        for(int i = node_cont - 2; i >= 0; i--) {
            // out:
            for(int s = node_cont - 1; s > i; s--) {
                for (int j = 0; j < cont_variables; j++) {
                    if (in[s][j] == 1) {
                        out[i][j] = 1;
                    }
                }
                if (kill[s] != '\0') out[i][hash_alphabet[kill[s] - 'a']] = 0;
            }
            // in:
            // out
            for(int j = 0; j < cont_variables; j++) {
                if(out[i][j] == 1) {
                    in[i][j] = 1;
                }
            }
            // (out[i] - kill[i])
            if(kill[i] != '\0'){
                in[i][hash_alphabet[kill[i] - 'a']] = 0;
            }
            
            // gen[i] U (out[i] - kill[i])
            int len = 0;
            if(list_of_nodes[i]->type == NODE_INT) len = list_of_nodes[i]->int_data.instruction->cont_operands;
            else if(list_of_nodes[i]->type == NODE_COND) len = list_of_nodes[i]->cond_data.cont_operands;
            else len = 0;
            for(int j = 0; j < len; j++) {
                if(gen[i][j] != '\0'){
                    in[i][hash_alphabet[gen[i][j] - 'a']] = 1;
                }
            }
        }
    } while(!(equal_ptrs(in, tmp_in, node_cont, cont_variables) && equal_ptrs(out, tmp_out, node_cont, cont_variables)));
    char* vars = (char*)malloc(sizeof(char) * cont_variables);
    for(int i = 0; i < cont_variables; i++) {
        for(int j = 0; j < 26; j++) {
            if(hash_alphabet[j] == i) vars[i] = j + 'a';
        }
    }

    if (debug_algorithm) {
        for(int i = 0; i < node_cont; i++) {
            for(int j = 0; j < cont_variables; j++) {
                printf("in[%i][%c] = %i, ", i + 1, vars[j], in[i][j]);
                
            }
            printf("\n");
            for(int j = 0; j < cont_variables; j++) {
                printf("out[%i][%c] = %i,", i + 1, vars[j], out[i][j]);
            }
            printf("\n");
        }
    }

    return 0;
}