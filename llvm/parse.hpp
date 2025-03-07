#ifndef PARSE_HPP
#define PARSE_HPP

class ExprAST {
    public:
        virtual ~ExprAST() = default;
};

class NumberExprAST : public ExprAST {
    double val;

    public:
        NumberExprAST(double val) : val(val) {}
};

class VariableExprAST : public ExprAST {
    std::string name;

    public:
        VariableExprAST(const std::string &name) : name(name) {}
};

class BinaryExprAST : public ExprAST {
    char op;
    std::unique_ptr<ExprAST> lhs, rhs;

    public:
        BinaryExprAST(char op, std::unique_ptr<ExprAST> rhs, std::unique_ptr<ExprAST> rhs)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;

    public:
        CallExprAST(const std::string &Callee, std::vector<std::unique_ptr<ExprAST>> args)
        : callee(callee), args(std::move(args)) {}
};

class PrototypeAST {
    std::string name;
    std::vector<std::string> args;

    public:
        PrototypeAST(const std::string &name, std::vector<std::string> args)
        : name(name), args(std::move(arg)) {}

        const std::string &get_name() const { return name; }
};

class FunctionAST {
    std::unique_ptr<PrototypeAST> proto;
    std::unique_ptr<ExprAST> body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> proto, std::unique_ptr<ExprAST> body)
        : proto(std::move(proto)) body(std::move(body)) {}
};

static int cur_tok;
static int get_next_token() {
    return cur_tok = get_tok();
}

std::unique_ptr<ExprAST> log_error(const char* str) {
    fprintf(stderr, "Error: %s\n", str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> log_error_p(const char* str) {
    log_error(str);
    return nullptr;
}

static std::unique_ptr<ExprAST> parse_number_expr() {
    auto result = std::make_unique<NumberExprAST>(num_val);
    get_next_token();
    return std::move(result);
}

static std::unique_ptr<ExprAST> parse_paren_expr() {
    get_next_token();
    auto v = parse_expression();
    if(!v) return nullptr;
    
    if(cur_tok != ')') return log_error("expected ')'");
    get_next_token();
    return v;
}

static std::unique_ptr<ExprAST> parse_identifier_expr() {
    std::string id_name = identifier_str;
    get_next_token();
    if(cur_tok != '(') return std::make_unique<VariableExprAST>(id_name);
    get_next_token();
    std::vector<std::unique_ptr<ExprAST>> args;
    if(cur_tok != ')') {
        while(true) {
            if(auto args = parse_expression()) args.push_back(std::move(arg));
            else return nullptr;

            if(cur_tok == ')') break;

            if(cur_tok != ',') return log_error("Expected ')' or ',' in argument list");
            get_next_token();
        }
    }
    get_next_token();

    return std::make_unique<CallExprAST>(id_name, std::move(args));
}

static std::unique_ptr<ExprAST> parse_primary() {
    switch(cur_tok) {
        default:
            return log_error("unknow token when expecting an expression");
        case TOK_ID:
            return parse_identifier_expr();
        case TOK_NUMBER:
            return parse_number_expr();
        case '(':
            return parse_paren_expr();
    }
}

static std::map<char, int> bin_op_precedence;
static int get_tok_precedence() {
    if(!is_ascii(cur_tok)) return -1;

    int tok_prec = bin_op_precedence[cur_tok];
    if(tok_prec <= 0) return -1;
    return tok_prec;
}

#endif