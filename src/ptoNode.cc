#include "ptoNode.hh"
#include "logger.hh"
#include <sstream>

namespace pto_parser {

PTO_EXPRESSION::PTO_EXPRESSION(const uint32_t& row, const uint32_t& col)
    : row_(row),
      col_(col)
{}


PTO_VARIABLE::PTO_VARIABLE(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      varName(name),
      varType(""),
      dataType(""),
      dimension(),
      typeStr()
{}

void PTO_VARIABLE::dump(int depth) const {
    SPDLOG_INFO("{}{}", std::string(depth * 2, ' '), varName);
    for (const auto& t : typeStr) {
        SPDLOG_INFO("{}type = {}", std::string(depth * 2, ' '), t);
    }
}

const std::string PTO_VARIABLE::to_string() const {
    return varName;
}

PTO_TUPLE_VAR::PTO_TUPLE_VAR(const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      varList()
{}

void PTO_TUPLE_VAR::dump(int depth) const {
    SPDLOG_INFO("{}Tuple of:", std::string(depth * 2, ' '));
    for (const auto& str : varList) {
        SPDLOG_INFO("{}{}", std::string(depth * 2 + 2, ' '), str);
    }
}

const std::string PTO_TUPLE_VAR::to_string() const {
    std::stringstream ss;
    for (const auto& str : varList) {
        ss << str << ", ";
    }
    return ss.str();
}

PTO_FLOAT::PTO_FLOAT(const float& v, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      value(v)
{}

void PTO_FLOAT::dump(int depth) const {
    SPDLOG_INFO("{}float value = {}", std::string(depth * 2, ' '), value);
}

const std::string PTO_FLOAT::to_string() const {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

PTO_INT::PTO_INT(const int& v, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      value(v)
{}

void PTO_INT::dump(int depth) const {
    SPDLOG_INFO("{}int value = {}", std::string(depth * 2, ' '), value);
}

const std::string PTO_INT::to_string() const {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

PTO_BOOL::PTO_BOOL(const bool& v, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      value(v)
{}

void PTO_BOOL::dump(int depth) const {
    SPDLOG_INFO("{}bool value = {}", std::string(depth * 2, ' '), value);
}

const std::string PTO_BOOL::to_string() const {
    if (value) return "true";
    else return "false";
}

PTO_BINARY_OP::PTO_BINARY_OP(PTO_OPERATOR o, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      lhs(nullptr),
      rhs(nullptr),
      op(o)
{}

PTO_BINARY_OP::~PTO_BINARY_OP() {
    if (lhs != nullptr) delete lhs;
    if (rhs != nullptr) delete rhs;
}

void PTO_BINARY_OP::dump(int depth) const {
    SPDLOG_INFO("{}{}", std::string(depth * 2, ' '), this->to_string());
}

const std::string PTO_BINARY_OP::to_string() const {
    std::stringstream ss;
    ss << "(";
    ss << lhs->to_string();
    switch (op) {
        case PTO_OPERATOR::ADD: ss << " + "; break;
        case PTO_OPERATOR::MUL: ss << " * "; break;
        case PTO_OPERATOR::FLOOR_DIV: ss << " // "; break;
        case PTO_OPERATOR::SUB: ss << " - "; break;
    }
    ss << rhs->to_string();
    ss << ")";
    return ss.str();
}

PTO_INDEXED_VAR::PTO_INDEXED_VAR(const std::string& n, const int& i, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      varName(n),
      index(std::vector<int>(1, i))
{}

void PTO_INDEXED_VAR::dump(int depth) const {
    if (index.size() != 1) {
        SPDLOG_ERROR("Unexpected index dimension for PTO_INDEXED_VAR");
    }
    SPDLOG_INFO("{}{}[{}]", std::string(depth * 2, ' '), varName, index[0]);
}

const std::string PTO_INDEXED_VAR::to_string() const {
    if (index.size() != 1) {
        SPDLOG_ERROR("Unexpected index dimension for PTO_INDEXED_VAR");
    }
    std::stringstream ss;
    ss << varName << "[" << index[0] << "]";
    return ss.str();
}

PTO_CALL::PTO_CALL(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      funcName(name),
      arguments()
{}

PTO_CALL::~PTO_CALL() {
    for (std::size_t i = 0; i < arguments.size(); i ++) {
        delete arguments[i];
    }
}

void PTO_CALL::dump(int depth) const {
    SPDLOG_INFO("{}{}(", std::string(depth * 2, ' '), funcName);
    for (const auto& arg : arguments) {
        arg->dump(depth + 1);
    }
    SPDLOG_INFO("{})", std::string(depth * 2, ' '));
}

const std::string PTO_CALL::to_string() const {
    std::string ret = funcName + "(";
    for (const auto& arg : arguments) {
        ret += arg->to_string() + ", ";
    }
    ret.pop_back();
    ret.back() = ')';
    return ret;
}

PTO_KEYWORD::PTO_KEYWORD(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_EXPRESSION(row, col),
      keyword(name),
      value(nullptr)
{}

PTO_KEYWORD::~PTO_KEYWORD() {
    if (value != nullptr) delete value;
}

void PTO_KEYWORD::dump(int depth) const {
    SPDLOG_INFO("{}{} =", std::string(depth * 2, ' '), keyword);
    
    if (value != nullptr)
        value->dump(depth + 1);
}

const std::string PTO_KEYWORD::to_string() const {
    return keyword + " = " + value->to_string();
}



PTO_BASE::PTO_BASE()
    : decorator_(),
      row_(),
      col_()
{}

PTO_BASE::PTO_BASE(const uint32_t& r, const uint32_t& c)
    : decorator_(),
      row_(r),
      col_(c)
{}


PTO_ASSIGNMENT::PTO_ASSIGNMENT(PTO_VARIABLE *l, const uint32_t& r, const uint32_t& c)
    : PTO_BASE(r, c),
      lhs(l),
      value(nullptr)
{}

PTO_ASSIGNMENT::~PTO_ASSIGNMENT() {
    if (lhs != nullptr)
        delete lhs;
    if (value != nullptr)
        delete value;
}

void PTO_ASSIGNMENT::dump(int depth) const {
    SPDLOG_INFO("{}ASSIGNMENT:", std::string(depth * 2, ' '));
    SPDLOG_INFO("{}lhs is:", std::string(depth * 2 + 2, ' '));
    lhs->dump(depth + 1);
    SPDLOG_INFO("{}rhs is:", std::string(depth * 2 + 2, ' '));
    value->dump(depth + 1);
}

PTO_RETURN::PTO_RETURN(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      returnVal()
{}

PTO_RETURN::~PTO_RETURN() {
    for (std::size_t i = 0; i < returnVal.size(); i ++) {
        delete returnVal[i];
    }
}

void PTO_RETURN::dump(int depth) const {
    SPDLOG_INFO("{}Return:", std::string(depth * 2, ' '));
    for (const auto& ptr : returnVal) {
        ptr->dump(depth + 1);
    }
}

PTO_FOR_LOOP::PTO_FOR_LOOP(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      iter(nullptr),
      initVar(),
      info(nullptr),
      statements()
{}

PTO_FOR_LOOP::~PTO_FOR_LOOP() {
    if (iter != nullptr)
        delete iter;
    for (std::size_t i = 0; i < initVar.size(); i ++) {
        delete initVar[i];
    }
    if (info != nullptr)
        delete info;
    for (std::size_t i = 0; i < statements.size(); i ++) {
        delete statements[i];
    }
}

void PTO_FOR_LOOP::dump(int depth) const {
    SPDLOG_INFO("{}For loop:", std::string(depth * 2, ' '));
    SPDLOG_INFO("{}Iterator:", std::string(depth * 2 + 2, ' '));
    iter->dump(depth + 2);
    SPDLOG_INFO("{}Init Variable", std::string(depth * 2 + 2, ' '));
    for (const auto& ptr : initVar) {
        ptr->dump(depth + 2);
    }
    info->dump(depth + 1);
    for (const auto& ptr : statements) {
        ptr->dump(depth + 1);
    }
}

PTO_FUNC::PTO_FUNC(const std::string& n, const uint32_t& r, const uint32_t& c)
    : PTO_BASE(r, c),
      funcName(n),
      decorate(),
      arguments(),
      returnTypeStr(),
      statements()
{}

PTO_FUNC::~PTO_FUNC() {
    for (std::size_t i = 0; i < arguments.size(); i ++) {
        delete arguments[i];
    }
    for (std::size_t i = 0; i < statements.size(); i ++) {
        delete statements[i];
    }
}

void PTO_FUNC::dump(int depth) const {
    SPDLOG_INFO("{}Decorate = {}", std::string(depth * 2, ' '), decorate);
    SPDLOG_INFO("{}Func Name = {}", std::string(depth * 2, ' '), funcName);
    SPDLOG_INFO("{}Input Parameters:", std::string(depth * 2, ' '));
    for (const auto& ptr : arguments) {
        ptr->dump(depth + 1);
    }
    if (returnTypeStr.size() == 1) {
        SPDLOG_INFO("{}Return Type = {}", std::string(depth * 2, ' '), returnTypeStr[0]);
    } else if (returnTypeStr.size() > 1) {
        SPDLOG_INFO("{}Return Type is tuple of:", std::string(depth * 2, ' '));
        for (const auto& str: returnTypeStr) {
            SPDLOG_INFO("{}  {}", std::string(depth * 2, ' '), str);
        }
    }
    for (const auto& ptr : statements) {
        ptr->dump(depth + 1);
    }
}

PTO_CLASS::PTO_CLASS(const std::string& n, const uint32_t& r, const uint32_t& c)
    : PTO_BASE(r, c),
      name(n),
      decorate(),
      functions()
{}

PTO_CLASS::~PTO_CLASS() {
    for (std::size_t i = 0; i < functions.size(); ++i)
        delete functions[i];
}

void PTO_CLASS::dump(int depth) const {
    SPDLOG_INFO("{} Decorate = {}", std::string(depth * 2, ' '), decorate);
    for (std::size_t i = 0; i < functions.size(); i ++) {
        functions[i]->dump(depth + 1);
    }
}

PTO_MODULE::PTO_MODULE()
    : classes(),
      functions(),
      globalVariable()
{}

PTO_MODULE::~PTO_MODULE() {
    for (std::size_t i = 0; i < classes.size(); ++i)
        delete classes[i];
    for (std::size_t i = 0; i < functions.size(); ++i)
        delete functions[i];
    for (auto& it : globalVariable) {
        delete it.second;
    }    
}


void PTO_MODULE::add_global_variable(PTO_ASSIGNMENT *assign) {
    globalVariable[assign->get_lhs()->to_string()] = assign;
}

void PTO_MODULE::add_class(PTO_CLASS *c) {
    classes.emplace_back(c);
}

void PTO_MODULE::dump(int depth) {
    SPDLOG_INFO("MODULE INFO");

    // 输出全局变量
    SPDLOG_INFO("{}Got {} global variables:", std::string(depth * 2, ' '), globalVariable.size());
    for (const auto& it : globalVariable) {
        SPDLOG_INFO("{}Variable Name = '{}'",  std::string(depth * 2 + 2, ' '), it.first);
        it.second->dump(depth + 1);
    }

    for (std::size_t i = 0; i < classes.size(); i ++) {
        classes[i]->dump(depth + 1);
    }
}

}