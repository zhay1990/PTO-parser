#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace pto_parser {
typedef std::unordered_map<std::string, std::string> STR_STR_MAP;
enum class PTO_EXPRESSION_TYPE{
    VARIABLE,
    TYPED_VARIABLE,
    CONSTANT,
    CALL,
    BINARY_OP,
    UNARY_OP,
    KEYWORD
};

class PTO_EXPRESSION{
public:
    PTO_EXPRESSION(const uint32_t& row, const uint32_t& col);
    virtual ~PTO_EXPRESSION() = default;

    virtual PTO_EXPRESSION_TYPE type() const = 0;

    virtual void dump(int depth) const = 0;
    virtual const std::string to_string() const = 0;

    const uint32_t& row() const {return row_;}
    const uint32_t& col() const {return col_;}
protected:
    uint32_t row_, col_;
};

class PTO_VARIABLE : public PTO_EXPRESSION {
public:
    explicit PTO_VARIABLE(const std::string& varName, const uint32_t& row, const uint32_t& col);
    ~PTO_VARIABLE();

    PTO_EXPRESSION_TYPE type() const {
        if (varType == "") return PTO_EXPRESSION_TYPE::VARIABLE;
        else              return PTO_EXPRESSION_TYPE::TYPED_VARIABLE;
    }
    void dump(int depth) const;
    const std::string to_string() const;

    void add_type_str(const std::string& str) {typeStr = str;}
private:
    std::string varName;
    
    std::string varType;

    // 数据类型
    std::string dataType;

    // 仅当类型是TENSOR时才生效的参数
    std::vector<int> dimension;

    // 解析时临时存储
    std::string typeStr;
};

class PTO_CALL : public PTO_EXPRESSION {
public:
    explicit PTO_CALL(const std::string& funcName, const uint32_t& row, const uint32_t& col);
    ~PTO_CALL();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::CALL;}
    void dump(int depth) const;
    const std::string to_string() const;

    
    void add_arguments(const std::vector<PTO_EXPRESSION*>& args) {arguments = args;}
private:
    std::string funcName;
    std::vector<PTO_EXPRESSION*> arguments;
};

class PTO_KEYWORD : public PTO_EXPRESSION {
public:
    explicit PTO_KEYWORD(const std::string& keyword, const uint32_t& row, const uint32_t& col);
    ~PTO_KEYWORD();

    PTO_EXPRESSION_TYPE type() const {return PTO_EXPRESSION_TYPE::KEYWORD;}
    void dump(int depth) const;
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
    ASSIGNMENT
};

class PTO_BASE {
public:
    PTO_BASE();
    PTO_BASE(const uint32_t& row, const uint32_t& col);
    virtual ~PTO_BASE() = default;

    virtual PTO_NODE_TYPE type() const = 0;
    virtual void dump(int depth) const = 0;


    const uint32_t& row() const {return row_;}
    const uint32_t& col() const {return col_;}
    const std::string& decorator() const {return decorator_;}

protected:
    std::string decorator_;
    uint32_t row_, col_;
};

class PTO_ASSIGNMENT : public PTO_BASE {
public:
    explicit PTO_ASSIGNMENT(const std::string& name, const uint32_t& row, const uint32_t& col);
    ~PTO_ASSIGNMENT();

    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::ASSIGNMENT;}
    void dump(int depth) const;

    void set_value(PTO_EXPRESSION* v) {value = v;}

    const std::string& get_lhs() const {return lhs;}

private:
    std::string lhs;
    PTO_EXPRESSION *value; // 可以是string，int，function call等等
};

class PTO_FUNC : public PTO_BASE {
public:
    explicit PTO_FUNC(const std::string& name, const uint32_t& row, const uint32_t& col);
    ~PTO_FUNC();

    void dump(int depth) const;
    PTO_NODE_TYPE type() const {return PTO_NODE_TYPE::FUNCTION;}

    void add_decoration(const std::string& d) {decorate = d;}
    void add_arguments(const std::vector<PTO_VARIABLE*>& arg) {arguments = arg;}
    void add_return_type_str(const std::vector<std::string>& str) {returnTypeStr = str;}
private:
    std::string funcName;
    std::string decorate;
    std::vector<PTO_VARIABLE*> arguments;
    std::vector<std::string> returnTypeStr;
};

class PTO_CLASS : public PTO_BASE{
public:
    explicit PTO_CLASS(const std::string& name, const uint32_t& row, const uint32_t& col);
    ~PTO_CLASS();

    void dump(int depth) const;
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

    void dump(int depth);

    void add_class(PTO_CLASS* c);
private:
    std::vector<PTO_CLASS*> classes;
    std::vector<PTO_FUNC*>  functions;
    std::unordered_map<std::string, PTO_ASSIGNMENT*> globalVariable;
};


} // namespace pto