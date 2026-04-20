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

} // namespace pto_parser
