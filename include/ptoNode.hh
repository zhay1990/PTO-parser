#pragma once

#include "ptoType.hh"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <fstream>

namespace pto_parser {
typedef std::unordered_map<std::string, std::string> STR_STR_MAP;

enum class PTO_NODE_TYPE {
    MODULE,
    CLASS,
    FUNCTION,
    
    // STATEMENT
    ASSIGNMENT,
    RETURN,
    FOR_LOOP,
    IF,
    
    // EXPRESSION
    VARIABLE,
    TYPED_VARIABLE,
    TUPLE_VARIABLE,
    LIST_VARIABLE,
    INDEXED_VARIABLE,
    FLOAT_CONSTANT,
    INT_CONSTANT,
    BOOL_CONSTANT,
    FUNC_CALL,
    KEYWORD,
    BINARY_OP
};

struct DEAD_CODE_RET{
    bool keepStatement;
    bool modified;

    DEAD_CODE_RET()
        : keepStatement(true),
          modified(false)
    {}
};

class PTO_BASE {
public:
    PTO_BASE(const uint32_t& row, const uint32_t& col);
    virtual ~PTO_BASE() = default;

    virtual PTO_NODE_TYPE type() const = 0;
    virtual void dump(int depth, std::ofstream& fout) const = 0;

    ///////////////////
    // 用于类型检查
    ///////////////////
    virtual bool type_check(std::unordered_map<std::string, PTO_TYPE>&) {return true;};
    virtual void infer_type(std::unordered_map<std::string, PTO_TYPE>&) {};

    ///////////////////
    // 用于死代码消除
    ///////////////////
    virtual bool get_callees(std::unordered_set<std::string>&) const {return true;}
    virtual bool add_to_live_map() const {return false;}
    virtual const struct DEAD_CODE_RET eliminate_dead_code() {return DEAD_CODE_RET();}
    // virtual void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const {}

    PTO_TYPE& get_data_type() {return dataType;}

    virtual const std::string to_string() const {return "";}

    const uint32_t& row() const {return row_;}
    const uint32_t& col() const {return col_;}
    const std::string& decorator() const {return decorator_;}

protected:
    std::string decorator_;
    uint32_t row_, col_;

    PTO_TYPE dataType;
};

class PTO_VARIABLE : public PTO_BASE {
public:
    explicit PTO_VARIABLE(const std::string& varName, const uint32_t& row, const uint32_t& col);
    ~PTO_VARIABLE() = default;

    PTO_NODE_TYPE type() const {
        if (typeStr.size() == 0) return PTO_NODE_TYPE::VARIABLE;
        else                     return PTO_NODE_TYPE::TYPED_VARIABLE;
    }
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool add_to_live_map() const override;

    void add_type_str(const std::string& str) {typeStr.emplace_back(str);}
    const std::vector<std::string>& get_type_str() const {return typeStr;}
private:
    std::string varName;
    // 解析时临时存储，多个typeStr表明该变量是tuple
    // 在当前处理的pyPTO文件里，tuple类型的变量都是函数调用时的赋值变量
    std::vector<std::string> typeStr;
};

class PTO_TUPLE_VAR : public PTO_BASE {
public:
    explicit PTO_TUPLE_VAR(const uint32_t& row, const uint32_t& col);
    ~PTO_TUPLE_VAR();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::TUPLE_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;
    
    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;

    void add_var(PTO_BASE* ptr) {varList.emplace_back(ptr);}

    const std::vector<PTO_BASE*>& get_var_list() const {return varList;}

    // void delete_dead_code(const std::vector<int>& deleteList);
private:
    std::vector<PTO_BASE*> varList;
};

class PTO_LIST_VAR : public PTO_BASE {
public:
    explicit PTO_LIST_VAR(const uint32_t& row, const uint32_t& col);
    ~PTO_LIST_VAR();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::LIST_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;

    void add_var(PTO_BASE* ptr) {varList.emplace_back(ptr);}

    const std::vector<PTO_BASE*>& get_var_list() const {return varList;}
private:
    std::vector<PTO_BASE*> varList;
};

class PTO_FLOAT : public PTO_BASE {
public:
    explicit PTO_FLOAT(const float& value, const uint32_t& row, const uint32_t& col);
    ~PTO_FLOAT() = default;
    
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FLOAT_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;

    const float& get_value() const {return value;}
private:
    float value;
};

class PTO_INT : public PTO_BASE {
public:
    explicit PTO_INT(const int& value, const uint32_t& row, const uint32_t& col);
    ~PTO_INT() = default;
    
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::INT_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;

    const int& get_value() const {return value;}
private:
    int value;
};

class PTO_BOOL : public PTO_BASE {
public:
    explicit PTO_BOOL(const bool& value, const uint32_t& row, const uint32_t& col);
    ~PTO_BOOL() = default;
    
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::BOOL_CONSTANT;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;

    const bool& get_value() const {return value;}
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
class PTO_BINARY_OP : public PTO_BASE {
public:
    explicit PTO_BINARY_OP(PTO_OPERATOR op, const uint32_t& row, const uint32_t& col);
    ~PTO_BINARY_OP();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::BINARY_OP;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;

    void set_lhs (PTO_BASE *l) {lhs = l;}
    void set_rhs (PTO_BASE *r) {rhs = r;}

private:
    PTO_BASE *lhs, *rhs;
    PTO_OPERATOR op;
};

class PTO_INDEXED_VAR : public PTO_BASE {
public:
    explicit PTO_INDEXED_VAR(const std::string& varName, const int& index, const uint32_t& row, const uint32_t& col);
    ~PTO_INDEXED_VAR() = default;

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::INDEXED_VARIABLE;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool add_to_live_map() const override;
private:
    std::string varName;
    // 这里给了拓展到多维的机会，但当前只处理tuple类型所以只有一个index
    std::vector<int> index;
};

class PTO_CALL : public PTO_BASE {
public:
    explicit PTO_CALL(const std::string& funcName, const uint32_t& row, const uint32_t& col);
    ~PTO_CALL();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FUNC_CALL;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
 
    void add_arguments(const std::vector<PTO_BASE*>& args) {arguments = args;}

    const std::string& get_func_name() const {return funcName;}
    const std::vector<PTO_BASE*>& get_arguments() const {return arguments;}

    // void delete_dead_code(const std::vector<int>& deleteList);
private:
    std::string funcName;
    std::vector<PTO_BASE*> arguments;
};

class PTO_KEYWORD : public PTO_BASE {
public:
    explicit PTO_KEYWORD(const std::string& keyword, const uint32_t& row, const uint32_t& col);
    ~PTO_KEYWORD();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::KEYWORD;}
    void dump(int depth, std::ofstream& fout) const;
    const std::string to_string() const;

    bool get_callees(std::unordered_set<std::string>&) const;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;
    
    void set_value(PTO_BASE* v) {value = v;}
    
    const std::string& get_keyword() const {return keyword;}
    const PTO_BASE* get_value() const {return value;}
private:
    std::string keyword;
    PTO_BASE* value;
};

class PTO_ASSIGNMENT : public PTO_BASE {
public:
    explicit PTO_ASSIGNMENT(PTO_BASE* lhs, const uint32_t& row, const uint32_t& col);
    ~PTO_ASSIGNMENT();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::ASSIGNMENT;}
    void dump(int depth, std::ofstream& fout) const;

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    const struct DEAD_CODE_RET eliminate_dead_code() override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;


    void set_value(PTO_BASE* v) {value = v;}

    PTO_BASE* get_lhs() const {return lhs;}
    PTO_BASE* get_value() const {return value;}

private:
    PTO_BASE *lhs;
    PTO_BASE *value; // 可以是string，int，function call等等
};

class PTO_RETURN : public PTO_BASE {
public:
    explicit PTO_RETURN(const uint32_t& row, const uint32_t& col);
    ~PTO_RETURN();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::RETURN;}
    void dump(int depth, std::ofstream& fout) const;

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    void infer_type(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;

    void add_value(PTO_BASE* v) {returnVal.emplace_back(v);}
    const std::vector<PTO_BASE*>& get_return_val() {return returnVal;}

    // void delete_dead_code(const std::unordered_set<int>& requriedReturn);
private:
    std::vector<PTO_BASE*> returnVal;
};

class PTO_FOR_LOOP : public PTO_BASE{
public:
    explicit PTO_FOR_LOOP(const uint32_t& row, const uint32_t& col);
    ~PTO_FOR_LOOP();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FOR_LOOP;}
    void dump(int depth, std::ofstream& fout) const;

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    bool add_to_live_map() const override;
    const struct DEAD_CODE_RET eliminate_dead_code() override;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;

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

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    // void collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>&) const;
    bool add_to_live_map() const override;
    const struct DEAD_CODE_RET eliminate_dead_code() override;
    
    
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

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;
    bool get_callees(std::unordered_set<std::string>&) const;
    const struct DEAD_CODE_RET eliminate_dead_code() override;


    void add_decoration(const std::string& d) {decorate = d;}
    void add_arguments(const std::vector<PTO_VARIABLE*>& arg) {arguments = arg;}
    void add_return_type_str(const std::vector<std::string>& str) {returnTypeStr = str;}
    void add_statement(PTO_BASE* statement) {statements.emplace_back(statement);}
    void add_statement(const std::vector<PTO_BASE*>& s) {statements.insert(statements.end(), s.begin(), s.end());}

    const std::string& get_func_name() const {return funcName;}
    const std::vector<std::string>& get_return_type() const {return returnTypeStr;}
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

    bool type_check(std::unordered_map<std::string, PTO_TYPE>& validVar) override;

    void add_decoration(const std::string& d) {decorate = d;}
    void add_function_def(PTO_FUNC* ptr) {functions.emplace_back(ptr);}

    const std::vector<PTO_FUNC*>& get_functions() const {return functions;}
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

    void add_class_or_func(PTO_BASE* c) {classOrFunc.emplace_back(c);};

    bool type_check() const;
    
    bool dead_code_eliminate();
private:
    std::vector<PTO_BASE*> classOrFunc;
    std::unordered_map<std::string, PTO_ASSIGNMENT*> globalVariable;
};


} // namespace pto