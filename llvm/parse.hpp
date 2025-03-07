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