#include "ptoNode.hh"
#include "logger.hh"

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
      typeStr("")
{}

PTO_VARIABLE::~PTO_VARIABLE() {}

void PTO_VARIABLE::dump(int depth) const {
    SPDLOG_INFO("{}{} {}", std::string(depth * 2, ' '), varName, typeStr);
}

const std::string PTO_VARIABLE::to_string() const {
    return varName;
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


PTO_ASSIGNMENT::PTO_ASSIGNMENT(const std::string& name, const uint32_t& r, const uint32_t& c)
    : PTO_BASE(r, c),
      lhs(name),
      value(nullptr)
{}

PTO_ASSIGNMENT::~PTO_ASSIGNMENT() {
    if (value != nullptr)
        delete value;
}

void PTO_ASSIGNMENT::dump(int depth) const {
    SPDLOG_INFO("{}ASSIGNMENT:", std::string(depth * 2, ' '));
    SPDLOG_INFO("{}lhs = {}", std::string(depth * 2 + 2, ' '), lhs);
    SPDLOG_INFO("{}rhs is:", std::string(depth * 2 + 2, ' '));
    value->dump(depth + 2);
}

PTO_FUNC::PTO_FUNC(const std::string& n, const uint32_t& r, const uint32_t& c)
    : PTO_BASE(r, c),
      funcName(n),
      decorate(),
      arguments(),
      returnTypeStr()
{}

PTO_FUNC::~PTO_FUNC() {

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
    globalVariable[assign->get_lhs()] = assign;
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