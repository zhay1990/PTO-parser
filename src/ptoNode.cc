#include "ptoNode.hh"
#include "logger.hh"
#include <sstream>

namespace pto_parser {
PTO_BASE::PTO_BASE(const uint32_t& r, const uint32_t& c)
    : decorator_(),
      row_(r),
      col_(c),
      dataType()
{}

// ==========================================
// PTO_VARIABLE
// ==========================================
PTO_VARIABLE::PTO_VARIABLE(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      varName(name),
      typeStr()
{}

void PTO_VARIABLE::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << varName;
    
    // 只输出typeStr
    if (typeStr.size() == 1) {    
        fout << " : " << typeStr[0];
    } else if (typeStr.size() > 1) {
        fout << " : pypto.language.Tuple([" << typeStr[0];
        for (std::size_t i = 1; i < typeStr.size(); ++i) {
            fout << ", " << typeStr[i];
        }
        fout << "])";
    }
}

const std::string PTO_VARIABLE::to_string() const {
    return varName;
}

// ==========================================
// PTO_TUPLE_VAR
// ==========================================
PTO_TUPLE_VAR::PTO_TUPLE_VAR(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      varList()
{}

PTO_TUPLE_VAR::~PTO_TUPLE_VAR() {
    for (std::size_t i = 0; i < varList.size(); ++i) {
        delete varList[i];
    }
}

void PTO_TUPLE_VAR::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << "(";
    if (varList.size() != 0) {
        varList[0]->dump(0, fout);
    }
    for (std::size_t i = 1; i < varList.size(); ++i) {
        fout << ", ";
        varList[i]->dump(0, fout);
    }
    fout << ")";
}

const std::string PTO_TUPLE_VAR::to_string() const {
    std::stringstream ss;
    ss << "(";
    if (varList.size() != 0) {
        ss << varList[0]->to_string();
    }
    for (std::size_t i = 1; i < varList.size(); ++i) {
        ss << ", ";
        ss << varList[i]->to_string();
    }
    ss << ")";
    return ss.str();
}

// ==========================================
// PTO_LIST_VAR
// ==========================================
PTO_LIST_VAR::PTO_LIST_VAR(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      varList()
{}

PTO_LIST_VAR::~PTO_LIST_VAR() {
    for (std::size_t i = 0; i < varList.size(); ++i) {
        delete varList[i];
    }
}

void PTO_LIST_VAR::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << "[";
    if (varList.size() != 0) {
        varList[0]->dump(0, fout);
    }
    for (std::size_t i = 1; i < varList.size(); ++i) {
        fout << ", ";
        varList[i]->dump(0, fout);
    }
    fout << "]";
}

const std::string PTO_LIST_VAR::to_string() const {
    std::stringstream ss;
    ss << "[";
    if (varList.size() != 0) {
        ss << varList[0]->to_string();
    }
    for (std::size_t i = 1; i < varList.size(); ++i) {
        ss << ", " << varList[i]->to_string();
    }
    ss << "]";
    return ss.str();
}

// ==========================================
// CONSTANTS (FLOAT, INT, BOOL)
// ==========================================
PTO_FLOAT::PTO_FLOAT(const float& v, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      value(v)
{}

void PTO_FLOAT::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << value;
}

const std::string PTO_FLOAT::to_string() const {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

PTO_INT::PTO_INT(const int& v, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      value(v)
{}

void PTO_INT::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << value;
}

const std::string PTO_INT::to_string() const {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

PTO_BOOL::PTO_BOOL(const bool& v, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      value(v)
{}

void PTO_BOOL::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    if (value) fout << indent << "True";
    else       fout << indent << "False";
}

const std::string PTO_BOOL::to_string() const {
    if (value) return "true";
    else return "false";
}

// ==========================================
// PTO_BINARY_OP
// ==========================================
PTO_BINARY_OP::PTO_BINARY_OP(PTO_OPERATOR o, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      lhs(nullptr),
      rhs(nullptr),
      op(o)
{}

PTO_BINARY_OP::~PTO_BINARY_OP() {
    if (lhs != nullptr) delete lhs;
    if (rhs != nullptr) delete rhs;
}

void PTO_BINARY_OP::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << '(';
    if (lhs != nullptr)
        lhs->dump(0, fout);
    
    switch (op) {
        case PTO_OPERATOR::ADD: fout << " + "; break;
        case PTO_OPERATOR::MUL: fout << " * "; break;
        case PTO_OPERATOR::FLOOR_DIV: fout << " // "; break;
        case PTO_OPERATOR::SUB: fout << " - "; break;
        case PTO_OPERATOR::EQUAL: fout << " == "; break;
    }
    if (rhs != nullptr)
        rhs->dump(0, fout);
    
    fout << ")";
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
        case PTO_OPERATOR::EQUAL: ss << " == "; break;
    }
    ss << rhs->to_string();
    ss << ")";
    return ss.str();
}

// ==========================================
// PTO_INDEXED_VAR
// ==========================================
PTO_INDEXED_VAR::PTO_INDEXED_VAR(const std::string& n, const int& i, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      varName(n),
      index(std::vector<int>(1, i))
{}

void PTO_INDEXED_VAR::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << varName << "[" << index[0] << "]";
}

const std::string PTO_INDEXED_VAR::to_string() const {
    if (index.size() != 1) {
        SPDLOG_ERROR("Unexpected index dimension for PTO_INDEXED_VAR");
    }
    std::stringstream ss;
    ss << varName << "[" << index[0] << "]";
    return ss.str();
}

// ==========================================
// PTO_CALL
// ==========================================
PTO_CALL::PTO_CALL(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      funcName(name),
      arguments()
{}

PTO_CALL::~PTO_CALL() {
    for (std::size_t i = 0; i < arguments.size(); i ++) {
        delete arguments[i];
    }
}

void PTO_CALL::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << funcName << "(";

    if (arguments.size() != 0) {
        arguments[0]->dump(0, fout);
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        fout << ", ";
        arguments[i]->dump(0, fout);
    }

    fout << ")";
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

// ==========================================
// PTO_KEYWORD
// ==========================================
PTO_KEYWORD::PTO_KEYWORD(const std::string& name, const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      keyword(name),
      value(nullptr)
{}

PTO_KEYWORD::~PTO_KEYWORD() {
    if (value != nullptr) delete value;
}

void PTO_KEYWORD::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent << keyword << " = ";
    if (value != nullptr) {
        value->dump(0, fout);
    }
}

const std::string PTO_KEYWORD::to_string() const {
    return keyword + " = " + value->to_string();
}

// ==========================================
// PTO_ASSIGNMENT
// ==========================================
PTO_ASSIGNMENT::PTO_ASSIGNMENT(PTO_BASE *l, const uint32_t& r, const uint32_t& c)
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

void PTO_ASSIGNMENT::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    fout << indent;

    if (lhs != nullptr) {
        lhs->dump(0, fout);
    }
    fout << " = ";
    if (value != nullptr) {
        value->dump(0, fout);
    }
}

// ==========================================
// PTO_RETURN
// ==========================================
PTO_RETURN::PTO_RETURN(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      returnVal()
{}

PTO_RETURN::~PTO_RETURN() {
    for (std::size_t i = 0; i < returnVal.size(); i ++) {
        delete returnVal[i];
    }
}

void PTO_RETURN::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    if (returnVal.size() != 0) {
        fout << indent << "return ";
        returnVal[0]->dump(0, fout);
        for (std::size_t i = 1; i < returnVal.size(); ++i) {
            fout << ", ";
            returnVal[i]->dump(0, fout);
        }
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

void PTO_FOR_LOOP::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');

    fout << indent << "for ";
    if (iter != nullptr) {
        iter->dump(0, fout);
    }
    fout << ", (";
    if (initVar.size() != 0) {
        initVar[0]->dump(0, fout);
    }
    for (std::size_t i = 1; i < initVar.size(); ++i) {
        fout << ", ";
        initVar[i]->dump(0, fout);
    }
    fout << ") in ";
    if (info != nullptr) {
        info->dump(0, fout);
    }
    fout << ":" << std::endl;

    for (const auto& ptr : statements) {
        ptr->dump(depth + 1, fout);
        fout << std::endl;
    }
}

PTO_IF::PTO_IF(const uint32_t& row, const uint32_t& col)
    : PTO_BASE(row, col),
      comparator(nullptr),
      ifStatement(),
      elseStatement()
{}

PTO_IF::~PTO_IF() {
    if (comparator != nullptr) delete comparator;
    for (std::size_t i = 0; i < ifStatement.size(); ++i) {
        delete ifStatement[i];
    }
    for (std::size_t i = 0; i < elseStatement.size(); ++i) {
        delete elseStatement[i];
    }
}

void PTO_IF::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');

    fout << indent << "if (";
    if (comparator != nullptr) {
        comparator->dump(0, fout);
    }
    fout << "):" << std::endl;
    for (const auto& ptr : ifStatement) {
        ptr->dump(depth + 1, fout);
        fout << std::endl;
    }
    if (elseStatement.size() != 0) {
        fout << indent << "else:" << std::endl;
    }
    for (const auto& ptr : elseStatement) {
        ptr->dump(depth + 1, fout);
        fout << std::endl;
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

void PTO_FUNC::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    if (decorate != "") {
        fout << indent << "@" << decorate << std::endl;
    }
    fout << indent << "def " << funcName << "(";

    if (arguments.size() != 0) {
        arguments[0]->dump(0, fout);
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        fout << ", ";
        arguments[i]->dump(0, fout);
    }
    fout << ")";

    if (returnTypeStr.size() == 1) {
        fout << " -> " << returnTypeStr[0];
    } else if (returnTypeStr.size() > 1) {
        fout << " -> " << "tuple[" << returnTypeStr[0];
        for (std::size_t i = 1; i < returnTypeStr.size(); ++i) {
            fout << ", " << returnTypeStr[i];
        }
        fout << "]";
    }
    fout << ":" << std::endl;

    for (const auto& ptr : statements){
        ptr->dump(depth + 1, fout);
        fout << std::endl;
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

void PTO_CLASS::dump(int depth, std::ofstream& fout) const {
    std::string indent = std::string(depth, '\t');
    if (decorate != "") {
        fout << indent << '@' << decorate << std::endl;
    }
    fout << indent << "class " << name << ":" << std::endl;
    for (const auto& ptr : functions) {
        ptr->dump(depth + 1, fout);
    }
}


PTO_MODULE::PTO_MODULE()
    : classOrFunc(),
      globalVariable()
{}

PTO_MODULE::~PTO_MODULE() {
    for (std::size_t i = 0; i < classOrFunc.size(); ++i)
        delete classOrFunc[i];
    for (auto& it : globalVariable) {
        delete it.second;
    }    
}


void PTO_MODULE::add_global_variable(PTO_ASSIGNMENT *assign) {
    globalVariable[assign->get_lhs()->to_string()] = assign;
}

void PTO_MODULE::dump(int depth, std::ofstream& fout) {
    std::string indent = std::string(depth, '\t');

    // 输出import
    fout << indent << "import pypto.language" << std::endl << std::endl;

    // 输出全局变量
    for (const auto& it : globalVariable) {
        it.second->dump(depth, fout);
        fout << std::endl;
    }
    fout << std::endl;

    for (const auto& ptr : classOrFunc) {
        ptr->dump(depth, fout);
    }
}
}