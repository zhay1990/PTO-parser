#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <fstream>

namespace pto_parser {
typedef std::unordered_map<std::string, std::string> STR_STR_MAP;
enum class PTO_EXPRESSION_TYPE{
    VARIABLE,
    TYPED_VARIABLE,
    TUPLE_VARIABLE,
    LIST_VARIABLE,
    INDEXED_VARIABLE,
    FLOAT_CONSTANT,
    INT_CONSTANT,
    BOOL_CONSTANT,
    CALL,
    KEYWORD,
    BINARY_OP
};

class PTO_EXPRESSION{
public:
    PTO_EXPRESSION(const uint32_t& row, const uint32_t& col);
    virtual ~PTO_EXPRESSION() = default;

    virtual PTO_EXPRESSION_TYPE type() const = 0;

    virtual void dump(int depth, std::ofstream& fout) const = 0;
    virtual const std::string to_string() const = 0;

    const uint32_t& row() const {return row_;}
    const uint32_t& col() const {return col_;}
protected:
    uint32_t row_, col_;
};

class PTO_VARIABLE : public PTO_EXPRESSION {
public:
    explicit PTO_VARIABLE(const std::string& varName, const uint32_t& row, const uint32_t& col);
    ~PTO_VARIABLE() = default;

    PTO_EXPRESSION_TYPE type() const {
        if (varType == "") return PTO_EXPRESSION_TYPE::VARIABLE;
        else               return PTO_EXPRESSION_TYPE::TYPED_VARIABLE;
    }
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void add_type_str(const std::string& str) {typeStr.emplace_back(str);}
private:
    std::string varName;
    
    std::string varType;

    // 数据类型
    std::string dataType;

    // 仅当类型是TENSOR时才生效的参数
    std::vector<int> dimension;

    // 解析时临时存储，多个typeStr表明该变量是tuple
    std::vector<std::string> typeStr;
};

class PTO_TUPLE_VAR : public PTO_EXPRESSION {
public:
    explicit PTO_TUPLE_VAR(const uint32_t& row, const uint32_t& col);
    ~PTO_TUPLE_VAR();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::TUPLE_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void add_var(PTO_EXPRESSION* ptr) {varList.emplace_back(ptr);}
private:
    std::vector<PTO_EXPRESSION*> varList;
};

class PTO_LIST_VAR : public PTO_EXPRESSION {
public:
    explicit PTO_LIST_VAR(const uint32_t& row, const uint32_t& col);
    ~PTO_LIST_VAR();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::LIST_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void add_var(PTO_EXPRESSION* ptr) {varList.emplace_back(ptr);}
private:
    std::vector<PTO_EXPRESSION*> varList;
};

class PTO_FLOAT : public PTO_EXPRESSION {
public:
    explicit PTO_FLOAT(const float& value, const uint32_t& row, const uint32_t& col);
    ~PTO_FLOAT() = default;
    
    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::FLOAT_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

private:
    float value;
};

class PTO_INT : public PTO_EXPRESSION {
public:
    explicit PTO_INT(const int& value, const uint32_t& row, const uint32_t& col);
    ~PTO_INT() = default;
    
    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::INT_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

private:
    int value;
};

class PTO_BOOL : public PTO_EXPRESSION {
public:
    explicit PTO_BOOL(const bool& value, const uint32_t& row, const uint32_t& col);
    ~PTO_BOOL() = default;
    
    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::BOOL_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

private:
    bool value;
};

enum class PTO_OPERATOR {
    ADD,
    MUL,
    SUB,
    FLOOR_DIV,
    EQUAL
};
class PTO_BINARY_OP : public PTO_EXPRESSION {
public:
    explicit PTO_BINARY_OP(PTO_OPERATOR op, const uint32_t& row, const uint32_t& col);
    ~PTO_BINARY_OP();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::BINARY_OP;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void set_lhs (PTO_EXPRESSION *l) {lhs = l;}
    void set_rhs (PTO_EXPRESSION *r) {rhs = r;}

private:
    PTO_EXPRESSION *lhs, *rhs;
    PTO_OPERATOR op;
};

class PTO_INDEXED_VAR : public PTO_EXPRESSION {
public:
    explicit PTO_INDEXED_VAR(const std::string& varName, const int& index, const uint32_t& row, const uint32_t& col);
    ~PTO_INDEXED_VAR() = default;

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::INDEXED_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

private:
    std::string varName;
    // 这里给了拓展到多维的机会，但当前只处理tuple类型所以只有一个index
    std::vector<int> index;
};

class PTO_CALL : public PTO_EXPRESSION {
public:
    explicit PTO_CALL(const std::string& funcName, const uint32_t& row, const uint32_t& col);
    ~PTO_CALL();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::CALL;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    
    void add_arguments(const std::vector<PTO_EXPRESSION*>& args) {arguments = args;}

    const std::string& get_func_name() const {return funcName;}
    const std::vector<PTO_EXPRESSION*>& get_arguments() const {return arguments;}
private:
    std::string funcName;
    std::vector<PTO_EXPRESSION*> arguments;
};

class PTO_KEYWORD : public PTO_EXPRESSION {
public:
    explicit PTO_KEYWORD(const std::string& keyword, const uint32_t& row, const uint32_t& col);
    ~PTO_KEYWORD();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::KEYWORD;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    
    void set_value(PTO_EXPRESSION* v) {value = v;}
private:
    std::string keyword;
    PTO_EXPRESSION* value;

};

enum class PTO_NODE_TYPE {
    MODULE,
    CLASS,
    FUNCTION,
    ASSIGNMENT,
    RETURN,
    FOR_LOOP,
    IF
};

class PTO_BASE {
public:
    PTO_BASE();
    PTO_BASE(const uint32_t& row, const uint32_t& col);
    virtual ~PTO_BASE() = default;

    virtual PTO_NODE_TYPE type() const = 0;
    virtual void dump(int depth, std::ofstream& fout) const = 0;


    const uint32_t& row() const {return row_;}
    const uint32_t& col() const {return col_;}
    const std::string& decorator() const {return decorator_;}

protected:
    std::string decorator_;
    uint32_t row_, col_;
};

class PTO_ASSIGNMENT : public PTO_BASE {
public:
    explicit PTO_ASSIGNMENT(PTO_VARIABLE* lhs, const uint32_t& row, const uint32_t& col);
    ~PTO_ASSIGNMENT();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::ASSIGNMENT;}
    void dump(int depth, std::ofstream& fout) const;

    void set_value(PTO_EXPRESSION* v) {value = v;}

    PTO_VARIABLE* get_lhs() const {return lhs;}

private:
    PTO_VARIABLE *lhs;
    PTO_EXPRESSION *value; // 可以是string，int，function call等等
};

class PTO_RETURN : public PTO_BASE {
public:
    explicit PTO_RETURN(const uint32_t& row, const uint32_t& col);
    ~PTO_RETURN();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::RETURN;}
    void dump(int depth, std::ofstream& fout) const;

    void add_value(PTO_EXPRESSION* v) {returnVal.emplace_back(v);}
private:
    std::vector<PTO_EXPRESSION*> returnVal;
};

class PTO_FOR_LOOP : public PTO_BASE{
public:
    explicit PTO_FOR_LOOP(const uint32_t& row, const uint32_t& col);
    ~PTO_FOR_LOOP();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FOR_LOOP;}
    void dump(int depth, std::ofstream& fout) const;

    void set_iterator(PTO_VARIABLE* it) {iter = it;}
    void set_call_info(PTO_CALL* c) {info = c;}

    void add_init_variable(PTO_VARIABLE* i) {initVar.emplace_back(i);}
    void add_init_variable(const std::vector<PTO_VARIABLE*>& i) {initVar.insert(initVar.end(), i.begin(), i.end());}


    void add_statement(PTO_BASE* statement) {statements.emplace_back(statement);}
    void add_statement(const std::vector<PTO_BASE*>& s) {statements.insert(statements.end(), s.begin(), s.end());}


private:
    PTO_VARIABLE *iter;
    std::vector<PTO_VARIABLE*> initVar;
    PTO_CALL* info;
    std::vector<PTO_BASE*> statements;
};

class PTO_IF : public PTO_BASE {
public:
    explicit PTO_IF(const uint32_t& row, const uint32_t& col);
    ~PTO_IF();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::IF;}
    void dump(int depth, std::ofstream& fout) const;

    void set_comparator(PTO_BINARY_OP *comp) {comparator = comp;}
    void add_if_statements(const std::vector<PTO_BASE*>& s) {ifStatement.insert(ifStatement.end(), s.begin(), s.end());}
    void add_else_statements(const std::vector<PTO_BASE*>& s) {elseStatement.insert(elseStatement.end(), s.begin(), s.end());}
private:
    PTO_BINARY_OP *comparator;
    std::vector<PTO_BASE*> ifStatement;
    std::vector<PTO_BASE*> elseStatement;
};

class PTO_FUNC : public PTO_BASE {
public:
    explicit PTO_FUNC(const std::string& name, const uint32_t& row, const uint32_t& col);
    ~PTO_FUNC();

    void dump(int depth, std::ofstream& fout) const;
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FUNCTION;}

    void add_decoration(const std::string& d) {decorate = d;}
    void add_arguments(const std::vector<PTO_VARIABLE*>& arg) {arguments = arg;}
    void add_return_type_str(const std::vector<std::string>& str) {returnTypeStr = str;}
    void add_statement(PTO_BASE* statement) {statements.emplace_back(statement);}
    void add_statement(const std::vector<PTO_BASE*>& s) {statements.insert(statements.end(), s.begin(), s.end());}

private:
    std::string funcName;
    std::string decorate;
    std::vector<PTO_VARIABLE*> arguments;
    std::vector<std::string> returnTypeStr;
    std::vector<PTO_BASE*> statements;
};

class PTO_CLASS : public PTO_BASE{
public:
    explicit PTO_CLASS(const std::string& name, const uint32_t& row, const uint32_t& col);
    ~PTO_CLASS();

    void dump(int depth, std::ofstream& fout) const;
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::CLASS;}

    void add_decoration(const std::string& d) {decorate = d;}
    void add_function_def(PTO_FUNC* ptr) {functions.emplace_back(ptr);}
private:
    std::string name;
    std::string decorate;
    std::vector<PTO_FUNC*> functions;
};

class PTO_MODULE {
public:
    explicit PTO_MODULE();
    ~PTO_MODULE();

    void add_global_variable(PTO_ASSIGNMENT* assign);

    void dump(int depth, std::ofstream& fout);

    void add_class(PTO_CLASS* c);
private:
    std::vector<PTO_CLASS*> classes;
    std::vector<PTO_FUNC*>  functions;
    std::unordered_map<std::string, PTO_ASSIGNMENT*> globalVariable;
};


} // namespace pto