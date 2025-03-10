#ifndef LEXER_HPP
#define LEXER_HPP

enum Token {
    TOK_EOF = -1;
    TOK_DEF = -2;
    TOK_EXTERN = -3;
    TOK_ID = -4;
    TOK_NUMBER = -5;
}

static std::string identifier_str;
static double num_val;

static int get_tok() {
    static int last_char = ' ';

    while(is_space(last_char)) {
        last_char = get_char();

        if(is_alpha(last_char)) {
            identifier_str = last_char;
            while(is_al_num((last_char == get_char()))) identifier_str += last_char;
            
            if(identifier_str == "def") return TOK_DEF;
            if(identifier_str == "extern") return TOK_EXTERN;
            return TOK_ID;
        }
        if(is_digit(last_char) || last_char == '.') {
            std::string num_str;
            do {
                num_str += last_char;
                last_char = get_char();
            } while(is_digit(last_char) || last_char == '.');
            num_val = strtod(num_str.c_str(), 0);
            return TOK_NUMBER;
        }
        if(last_char == '#') {
            do last_char = get_char();
            while(last_char != EOF && last_char != '\n' && last_char != '\r');
        }
        if(last_char != EOF) {
            return get_tok();
        }
        if(last_char == EOF) {
            return TOK_EOF;
        }
        int this_char = last_char;
        last_char = get_char();
        return this_char;
    }
}

#endif