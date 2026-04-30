#include "ptoNode.hh"
#include "logger.hh"

namespace pto_parser
{
////////////////////////////////////////
// 使用并查集消除赋值后产生的等价变量名
////////////////////////////////////////

// 并查集辅助结构
static std::unordered_map<std::string, std::string> unionParent;

const std::string find_var(const std::string& varName) {
    // 如果第一次见到这个变量名，则返回它自己
    if (unionParent.find(varName) == unionParent.end()) {
        unionParent[varName] = varName;
    }

    // 寻找到了根节点
    if (unionParent[varName] == varName) {
        return varName;
    }

    // 路径压缩：直接把当前节点连到终极老大上，加速后续查找
    return unionParent[varName] = find_var(unionParent[varName]);
}

void union_var(const std::string& lhs, const std::string& rhs) {
    const std::string rootLhs = find_var(lhs);
    const std::string rootRhs = find_var(rhs);

    if (rootLhs != rootRhs) {
        // 先比长度再比字典序
        if (rootLhs.size() < rootRhs.size()) {
            unionParent[rootRhs] = rootLhs;
        } 
        else if (rootLhs.size() == rootRhs.size() && rootLhs < rootRhs){
            unionParent[rootRhs] = rootLhs;
        }
        else {
            unionParent[rootLhs] = rootRhs;
        }
    }
}


bool PTO_MODULE::alias_coalasce() const {
    // 简化处理，假定一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return false;
    }

    return classOrFunc[0]->alias_coalasce();
}

bool PTO_CLASS::alias_coalasce() {
    bool ret = false;
    for (auto& func : functions) {
        ret |= func->alias_coalasce();
    }

    return ret;
}

bool PTO_FUNC::alias_coalasce() {
    // 初始化并查集
    unionParent.clear();

    // 构建并查集
    for (const auto& s : statements) {
        s->build_alias_union();
    }

    // 基于并查集，重命名变量
    bool ret = false;
    for (auto& s : statements) {
        ret |= s->alias_coalasce();
    }

    // 还需要重命名argmuments
    for (auto& arg : arguments) {
        ret |= arg->alias_coalasce();
    }

    return ret;
}

void PTO_ASSIGNMENT::build_alias_union() const {
    // 处理纯变量赋值
    if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE && value->type() == PTO_NODE_TYPE::VARIABLE) {
        union_var(lhs->to_string(), value->to_string());
        SPDLOG_DEBUG("Variable {} and {} at line {} are unioned", lhs->to_string(), value->to_string(), row_);
    }
}

void PTO_FOR_LOOP::build_alias_union() const {
    for (const auto& s : statements) {
        s->build_alias_union();
    }
}

void PTO_IF::build_alias_union() const {
    // 不处理if语句
    // 在标准的条件分支中，如果存在 if cond: out = A 和 else: out = B 的结构，
    // 无脑的并查集会将 out 分别与 A 和 B 连通，
    // 最终导致原本互斥的 A 和 B 被错误地合并为同一个变量（Transitive Union 污染）。
}

bool PTO_VARIABLE::alias_coalasce() {
    if (find_var(varName) == varName) return false;

    SPDLOG_DEBUG("Variable {} at line {} is changed to {}", varName, row_, find_var(varName));

    varName = find_var(varName);
    return true;
}

bool PTO_ASSIGNMENT::alias_coalasce() {
    bool ret = value->alias_coalasce();

    ret |= lhs->alias_coalasce();

    return ret;
}

bool PTO_IF::alias_coalasce() {
    bool ret = false;
    ret |= comparator->alias_coalasce();

    for (auto& s : ifStatement) {
        ret |= s->alias_coalasce();
    }

    for (auto& s : elseStatement) {
        ret |= s->alias_coalasce();
    }

    return ret;
}

bool PTO_FOR_LOOP::alias_coalasce() {
    bool ret = info->alias_coalasce();
    for (auto& s : statements) {
        ret |= s->alias_coalasce();
    }

    return ret;
}

bool PTO_RETURN::alias_coalasce() {
    bool ret = false;
    for (auto& val : returnVal) {
        if (val->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            continue;
        }

        ret |= val->alias_coalasce();
    }

    return ret;
}

bool PTO_CALL::alias_coalasce() {
    bool ret = false;
    for (auto& s : arguments) {
        ret |= s->alias_coalasce();
    }
    return ret;
}

bool PTO_TUPLE_VAR::alias_coalasce() {
    bool ret = false;
    for (auto& s : varList) {
        ret |= s->alias_coalasce();
    }
    return ret;
}

bool PTO_LIST_VAR::alias_coalasce() {
    bool ret = false;
    for (auto& s : varList) {
        ret |= s->alias_coalasce();
    }
    return ret;
}

bool PTO_INDEXED_VAR::alias_coalasce() {
    if (find_var(varName) == varName) return false;

    SPDLOG_DEBUG("Indexed variable {} at line {} is changed to {}", varName, row_, find_var(varName));
    varName = find_var(varName);

    return true;
}

bool PTO_BINARY_OP::alias_coalasce() {
    bool ret = false;
    ret |= lhs->alias_coalasce();
    ret |= rhs->alias_coalasce();
    return ret;
}

bool PTO_KEYWORD::alias_coalasce() {
    return value->alias_coalasce();
}

//////////////////////////////////////////
// 构建kernel函数输入输出的等价关系
//////////////////////////////////////////

// 记录output到input的映射关系
static std::unordered_map<std::string, std::unordered_map<int, int>> funcOut2InMap;
static std::unordered_map<int, int> curOut2InMap;
// 记录当前函数的输入名字和顺序
static std::unordered_map<std::string, int> curInputMap;



void PTO_MODULE::func_input_output_coalasce() const {
    // 简化处理，假定一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return;
    }

    classOrFunc[0]->func_input_output_coalasce();
}

void PTO_CLASS::func_input_output_coalasce() {
    for (auto& func : functions) {
        func->func_input_output_coalasce();
    }
}

static void create_assignment(PTO_ASSIGNMENT *assignPtr, std::vector<PTO_BASE*>& statements, int targetIndex) {
    // 用户自定义函数调用?
    if (assignPtr->get_value()->type() != PTO_NODE_TYPE::FUNC_CALL) {
        return;
    }

    auto funcPtr = (PTO_CALL*)assignPtr->get_value();
    if (funcPtr->get_func_name().substr(0, 5) != "self.") {
        return;
    }

    if (funcOut2InMap.find(funcPtr->get_func_name().substr(5)) == funcOut2InMap.end()) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    const auto& args = funcPtr->get_arguments();

    // 根据output到input的映射关系，添加assign语句
    for (const auto& it : funcOut2InMap[funcPtr->get_func_name().substr(5)]) {
        // 找到lhs里第it.first的变量名
        PTO_VARIABLE *lhs = nullptr, *rhs = nullptr;

        auto lhsPtr = assignPtr->get_lhs();

        if (lhsPtr->type() == PTO_NODE_TYPE::VARIABLE || lhsPtr->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            // 左侧只有一个输出
            if (it.first != 0) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }
            lhs = new PTO_VARIABLE(lhsPtr->to_string(), lhsPtr->row(), lhsPtr->col());
            lhs->add_type_str(((PTO_VARIABLE*)lhsPtr)->get_type_str()[0]);
        }
        else if (lhsPtr->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            const auto& varList = ((PTO_TUPLE_VAR*)lhsPtr)->get_var_list();
            if (it.first >= (int)varList.size()) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            lhs = new PTO_VARIABLE(varList[it.first]->to_string(), varList[it.first]->row(), varList[it.first]->col());
            lhs->add_type_str(((PTO_VARIABLE*)varList[it.first])->get_type_str()[0]);
        }
        else {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (it.second >= (int)args.size()) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (args[it.second]->type() == PTO_NODE_TYPE::VARIABLE) {
            rhs = new PTO_VARIABLE(args[it.second]->to_string(), args[it.second]->row(), args[it.second]->col());
        }
        else {
            SPDLOG_ERROR("Unimplemented");
            return;
        }

        auto newAssign = new PTO_ASSIGNMENT(lhs, lhs->row(), lhs->col());
        newAssign->set_value(rhs);

        statements.insert(statements.begin() + targetIndex, newAssign);
    }
}

void PTO_FUNC::func_input_output_coalasce() {
    // 先记录入参的顺序信息
    curInputMap.clear();
    if (arguments[0]->to_string() != "self") {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    for (std::size_t i = 0; i < arguments.size(); ++i) {
        curInputMap[arguments[i]->to_string()] = i - 1;
    }

    // 扫描所有语句，更新funcOut2InMap，并且根据已有的funcOut2InMap添加映射语句
    for (std::size_t i = 0; i < statements.size(); ++i) {
        if (statements[i]->type() == PTO_NODE_TYPE::ASSIGNMENT) {
            create_assignment((PTO_ASSIGNMENT*)statements[i], statements, i + 1);
        }
        else {
            statements[i]->func_input_output_coalasce();
        }
    }

    funcOut2InMap[funcName] = curOut2InMap;
    curOut2InMap.clear();
}

void PTO_IF::func_input_output_coalasce() {
    for (std::size_t i = 0; i < ifStatement.size(); ++i) {
        if (ifStatement[i]->type() == PTO_NODE_TYPE::ASSIGNMENT) {
            create_assignment((PTO_ASSIGNMENT*)ifStatement[i], ifStatement, i + 1);
        } else {
            ifStatement[i]->func_input_output_coalasce();
        }
    }
    for (std::size_t i = 0; i < elseStatement.size(); ++i) {
        if (elseStatement[i]->type() == PTO_NODE_TYPE::ASSIGNMENT) {
            create_assignment((PTO_ASSIGNMENT*)elseStatement[i], elseStatement, i + 1);
        } else {
            elseStatement[i]->func_input_output_coalasce();
        }
    }
}

void PTO_FOR_LOOP::func_input_output_coalasce() {
    for (std::size_t i = 0; i < statements.size(); ++i) {
        if (statements[i]->type() == PTO_NODE_TYPE::ASSIGNMENT) {
            create_assignment((PTO_ASSIGNMENT*)statements[i], statements, i + 1);
        } else {
            statements[i]->func_input_output_coalasce();
        }
    }
}

void PTO_RETURN::func_input_output_coalasce() {
    // 这里需要构建output到input的映射关系
    for (std::size_t i = 0; i < returnVal.size(); ++i) {
        if (returnVal[i]->type() == PTO_NODE_TYPE::VARIABLE) {
            if (curInputMap.find(returnVal[i]->to_string()) == curInputMap.end()) {
                continue;
            }
            if (curOut2InMap.find(i) != curOut2InMap.end() && curOut2InMap[i] != curInputMap[returnVal[i]->to_string()]) {
                SPDLOG_ERROR("Unimplemented");
                return;
            }
            curOut2InMap[i] = curInputMap[returnVal[i]->to_string()];
        } else {
            SPDLOG_ERROR("Unimplemented");
            return;
        }
    }
}


} // namespace pto_parser
