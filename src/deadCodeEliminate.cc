#include "ptoNode.hh"
#include "logger.hh"
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace pto_parser {

struct CallGraphNode {
    PTO_FUNC *func;
    int in_degree = 0;
    std::vector<CallGraphNode*> callees;
    bool isRoot;
};

// 记录所有函数哪些输出是需要的
static std::unordered_map<std::string, std::unordered_set<int>> requiredReturnIndices;
// 记录所有函数哪些输入被删除
static std::unordered_map<std::string, std::vector<int>> removedFuncInput;
// 记录所有函数哪些输出被删除
static std::unordered_map<std::string, std::vector<int>> removedFuncOutput;

// 记录当前函数的哪些输出是需要的，在处理return statement时使用
static std::unordered_set<int> curRequiredReturnIndices;
// 记录当前函数的有效变量，因为不存在index var了，所以只需要记录变量名
static std::unordered_set<std::string> curLiveMap;


typedef std::unordered_map<std::string, CallGraphNode*> STR_GRAPH_MAP;
typedef std::unordered_map<std::string, struct LIVE_VARIABLE> STR_LIVE_MAP;

// 注意！！
// 在调用这个函数前，应当先调用remove_yield和remove_indexed_var函数
bool PTO_MODULE::dead_code_eliminate() {
    // 先构建调用关系图
    std::unordered_map<std::string, CallGraphNode*> funcInfo;

    // 简化处理，假定一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return false;
    }

    // 拿到这个class的函数定义
    auto funcPtrs = ((PTO_CLASS*)classOrFunc[0])->get_functions();

    for (const auto& func : funcPtrs) {
        // 扫描这个函数包含的statement，确认这个函数调用了哪些函数
        std::unordered_set<std::string> callees;
        if (!func->get_callees(callees))
            return false;
        
        // 根据callees构建funcInfo
        auto ptr = new struct CallGraphNode();
        ptr->func = func;
        ptr->isRoot = false;
        
        for (const auto& c : callees) {
            if (funcInfo.find(c) == funcInfo.end()) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }
            funcInfo[c]->in_degree += 1;
            ptr->callees.emplace_back(funcInfo[c]);
        }
        funcInfo[func->get_func_name()] = ptr;
    }

    // 先确认root函数，root函数的所有输出都是需要的
    for (const auto& it : funcInfo) {
        if (it.second->in_degree == 0) {
            it.second->isRoot = true;
        }
    }

    // 确认处理顺序
    std::vector<CallGraphNode*> processList;
    while (processList.size() != funcInfo.size()) {
        for (const auto& it : funcInfo) {
            if (it.second->in_degree != 0) continue;
            processList.emplace_back(it.second);

            for (const auto& c : it.second->callees) {
                c->in_degree -= 1;
            }

            it.second->in_degree -= 1;
        }
    }

    // 初始化为空列表
    for (const auto& p : processList) {
        requiredReturnIndices[p->func->get_func_name()] = std::unordered_set<int>();
        removedFuncInput[p->func->get_func_name()] = std::vector<int>();
        removedFuncOutput[p->func->get_func_name()] = std::vector<int>();
    }

    // 循环处理，逐步消除死代码
    while (true) {
        // 清空需要的返回值
        for (auto& it : requiredReturnIndices) {
            it.second.clear();
        }

        // root函数的所有输出都需要
        for (const auto& p : processList) {
            if (p->isRoot) {
                requiredReturnIndices[p->func->get_func_name()].insert(-1);
            }
        }

        // 初始化函数返回和输入值的删除记录
        for (auto& it : removedFuncInput) {
            it.second.clear();
        }
        for (auto& it : removedFuncOutput) {
            it.second.clear();
        }

        // 先消除死代码
        bool eliminated = false;
        for (const auto& p : processList) {
            curRequiredReturnIndices = requiredReturnIndices[p->func->get_func_name()];
            curLiveMap.clear();

            if (curRequiredReturnIndices.size() == 0) {
                // 这个函数没有有效输出，不做任何处理
                SPDLOG_WARN("Function {} at line {} is not called by any other functions", p->func->get_func_name(), p->func->row());
                continue;
            }

            const auto ret = p->func->eliminate_dead_code();
            eliminated |= ret.modified;
        }

        // 如果没有死代码被消除则退出循环
        if (!eliminated) break;

        // 有死代码消除的情况下，需要再顺序检查一遍，修改函数调用的入参和返回值
        for (const auto& p : processList) {
            if (requiredReturnIndices[p->func->get_func_name()].size() == 0) {
                // 不处理无用函数
                continue;
            }
            p->func->adjust_user_func_input();
        }
    }

    // clean up
    for (auto& it : funcInfo) {
        delete it.second;
    }
    funcInfo.clear();
    return true;
}

bool PTO_FUNC::get_callees(std::unordered_set<std::string>& callees) const {
    for (const auto& s : statements) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}

bool PTO_IF::get_callees(std::unordered_set<std::string>& callees) const {
    if (!comparator->get_callees(callees))
        return false;
    for (const auto& s : ifStatement) {
        if (!s->get_callees(callees))
            return false;
    }
    for (const auto& s : elseStatement) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}

bool PTO_FOR_LOOP::get_callees(std::unordered_set<std::string>& callees) const {
    for (const auto& s : statements) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}

bool PTO_RETURN::get_callees(std::unordered_set<std::string>& callees) const {
    for (const auto& s : returnVal) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}

bool PTO_ASSIGNMENT::get_callees(std::unordered_set<std::string>& callees) const {
    return value->get_callees(callees);
}

bool PTO_KEYWORD::get_callees(std::unordered_set<std::string>& callees) const {
    return value->get_callees(callees);
}

bool PTO_CALL::get_callees(std::unordered_set<std::string>& callees) const {
    // 只记录self.xxx函数
    if (funcName.substr(0, 5) == "self.") {
        callees.insert(funcName.substr(5));
    }
    return true;
}

bool PTO_BINARY_OP::get_callees(std::unordered_set<std::string>& callees) const {
    return lhs->get_callees(callees) && rhs->get_callees(callees);
}

bool PTO_LIST_VAR::get_callees(std::unordered_set<std::string>& callees) const {
    for (const auto& s : varList) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}

bool PTO_TUPLE_VAR::get_callees(std::unordered_set<std::string>& callees) const {
    for (const auto& s : varList) {
        if (!s->get_callees(callees))
            return false;
    }
    return true;
}


const struct DEAD_CODE_RET PTO_FUNC::eliminate_dead_code() {
    SPDLOG_DEBUG("Processing func {} for dead code elimination", funcName);

    // 边构建liveMap 边处理statements的删除
    struct DEAD_CODE_RET ret;
    for (int i = (int)statements.size() - 1; i >= 0; --i) {
        statements[i]->add_to_live_map();

        auto temp = statements[i]->eliminate_dead_code();

        if (!temp.keepStatement) {
            SPDLOG_DEBUG("Statement at line {} is removed", statements[i]->row());
            ret.modified = true;
            delete statements[i];
            statements[i] = nullptr;
        } else {
            ret.modified |= temp.modified;
        }
    }
    
    for (std::size_t i = 0; i < statements.size();) {
        if (statements[i] == nullptr) {
            statements.erase(statements.begin() + i);
        } else {
            ++i;
        }
    }
    
    // 修改返回值，记录哪些输出被删除
    if (curRequiredReturnIndices.find(-1) == curRequiredReturnIndices.end()) {
        for (std::size_t i = 0; i < returnTypeStr.size(); ++i) {
            if (curRequiredReturnIndices.find(i) == curRequiredReturnIndices.end()) {
                SPDLOG_DEBUG("Return type {} is removed for func {} at line {}", returnTypeStr[i], funcName, row_);
                returnTypeStr[i] = "";
                ret.modified = true;
                removedFuncOutput[funcName].emplace_back(i);
            }
        }
    }

    for (std::size_t i = 0; i < returnTypeStr.size();) {
        if (returnTypeStr[i] == "") {
            returnTypeStr.erase(returnTypeStr.begin() + i);
        } else {
            ++ i;
        }
    }

    // 修改入参，记录哪些输入被删除
    // 默认用户自定义函数的第一个参数是self
    if (arguments[0]->to_string() != "self") {
        SPDLOG_ERROR("The first argument of user defined function should be self");
        return ret;
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        if (curLiveMap.find(arguments[i]->to_string()) == curLiveMap.end()) {
            SPDLOG_DEBUG("Input argument {} is removed for func {} at line {}", arguments[i]->to_string(), funcName, row_);
            // 在函数调用时，self不会作为入参，所以需要-1
            removedFuncInput[funcName].emplace_back(i - 1);
            delete arguments[i];
            arguments[i] = nullptr;
            ret.modified = true;
        }
    }

    for (std::size_t i = 0; i < arguments.size();) {
        if (arguments[i] == nullptr) {
            arguments.erase(arguments.begin() + i);
        } else {
            ++i;
        }
    }

    return ret;
}

bool PTO_RETURN::add_to_live_map() const {
    // 基于curRequiredReturnIndices添加live variable
    bool ret = false;
    if (curRequiredReturnIndices.find(-1) != curRequiredReturnIndices.end()) {
        // 这个函数的所有输出都是需要的
        for (const auto& val : returnVal) {
            ret |= val->add_to_live_map();
        }
        return ret;
    }

    for (std::size_t i = 0; i < returnVal.size(); ++i) {
        if (curRequiredReturnIndices.find(i) != curRequiredReturnIndices.end()) {
            ret |= returnVal[i]->add_to_live_map();
        }
    }
    return ret;
}

bool PTO_FOR_LOOP::add_to_live_map() const {
    // remove_yield函数应当清空了initVar
    if (initVar.size() != 0) {
        SPDLOG_ERROR("Should call remove_yield() before calling eliminate_dead_code()");
        return false;
    }
    
    // 循环处理，确定所有的live variable
    bool ret = false;
    while (true) {
        bool varAdded = false;

        // 倒序处理所有语句
        for (int i = (int)statements.size() - 1; i >= 0; --i) {
            varAdded |= statements[i]->add_to_live_map();
        }

        // 注意 info里的第一个变量也是需要的
        info->get_arguments()[0]->add_to_live_map();

        if (!varAdded) break;

        ret = true;
    }

    // info里的init values应当是空的
    for (const auto& arg : info->get_arguments()) {
        if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "init_values") {
            const auto& tuple = ((PTO_KEYWORD*)arg)->get_value();
            if (tuple->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return ret;
            }

            const auto& varList = ((PTO_TUPLE_VAR*)tuple)->get_var_list();
            if (varList.size() != 0) {
                SPDLOG_ERROR("Should call remove_yield() before calling eliminate_dead_code()");
                return false;
            }
        }
    }

    return ret;
}

bool PTO_VARIABLE::add_to_live_map() const {
    if (curLiveMap.find(varName) == curLiveMap.end()) {
        curLiveMap.insert(varName);
        SPDLOG_DEBUG("Variable {} at line {} added to live map", varName, row_);
        return true;
    }
    return false;
}

bool PTO_ASSIGNMENT::add_to_live_map() const {
    bool ret = false;

    // 应当不存在yield语句
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.yield_") {
        SPDLOG_ERROR("Should call remove_yield() before calling eliminate_dead_code()");
        return false;
    }

    if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (curLiveMap.find(lhs->to_string()) != curLiveMap.end()) {
            // 左值被需要
            ret |= value->add_to_live_map();

            // 右值是用户自定义函数？
            if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name().substr(0, 5) == "self.") {

                std::string userFuncName = ((PTO_CALL*)value)->get_func_name().substr(5);

                // 这个函数的所有输出都需要，插入-1表示该信息（实质上只有第一个输出需要）
                requiredReturnIndices[userFuncName].insert(-1);
                SPDLOG_DEBUG("All outputs of function {} are required from line {}", userFuncName, row_);
            }
        }
    }
    else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
        // 在这种情况下，要求右侧必须是用户自定义函数调用
        if (value->type() != PTO_NODE_TYPE::FUNC_CALL || ((PTO_CALL*)value)->get_func_name().substr(0, 5) != "self.") {
            SPDLOG_ERROR("Should be user function call at line {}", row_);
            return false;
        }

        const auto& varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
        std::string userFuncName = ((PTO_CALL*)value)->get_func_name().substr(5);

        bool outputIsNeeded = false;

        for (std::size_t i = 0; i < varList.size(); ++i) {
            if (varList[i]->type() != PTO_NODE_TYPE::VARIABLE && varList[i]->type() != PTO_NODE_TYPE::TYPED_VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }

            if (curLiveMap.find(varList[i]->to_string()) == curLiveMap.end()) {
                continue;
            }

            outputIsNeeded = true;

            // 右侧函数的第i个输出是需要的
            requiredReturnIndices[userFuncName].insert(i);
            SPDLOG_DEBUG("The [{}] outputs of function {} are required from line {}", i, userFuncName, row_);
        }

        if (outputIsNeeded) {
            ret |= value->add_to_live_map();
        }
    }
    else {
        SPDLOG_ERROR("Unexpected Error");
        return false;
    }

    return ret;
}

bool PTO_INDEXED_VAR::add_to_live_map() const {
    // 不应该存在index var
    SPDLOG_ERROR("Should call remove_indexed_var() before calling eliminate_dead_code()");
    return false;
}

bool PTO_CALL::add_to_live_map() const {
    // 不会出现yield
    if (funcName == "pypto.language.yield_") {
        SPDLOG_ERROR("Should call remove_yield() before calling eliminate_dead_code()");
        return false;
    }

    // 将arguments添加到live map
    bool ret = false;
    for (const auto& arg : arguments) {
        ret |= arg->add_to_live_map();
    }
    return ret;
}

bool PTO_BINARY_OP::add_to_live_map() const {
    bool ret = lhs->add_to_live_map();
    ret |= rhs->add_to_live_map();

    return ret;
}

bool PTO_TUPLE_VAR::add_to_live_map() const {
    // 将arguments添加到live map
    bool ret = false;
    for (const auto& arg : varList) {
        ret |= arg->add_to_live_map();
    }
    return ret;
}

bool PTO_LIST_VAR::add_to_live_map() const {
    // 将arguments添加到live map
    bool ret = false;
    for (const auto& arg : varList) {
        ret |= arg->add_to_live_map();
    }
    return ret;
}

bool PTO_IF::add_to_live_map() const {
    bool ret = comparator->add_to_live_map();

    for (int i = (int)ifStatement.size() - 1; i >= 0; --i) {
        ret |= ifStatement[i]->add_to_live_map();
    }
    for (int i = (int)elseStatement.size() - 1; i >= 0; --i) {
        ret |= elseStatement[i]->add_to_live_map();
    }
    
    return ret;
}

const struct DEAD_CODE_RET PTO_RETURN::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    // 根据liveMap修改returnVal列表
    for (std::size_t i = 0; i < returnVal.size(); ++i) {
        if (returnVal[i]->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return ret;
        }
        if (curLiveMap.find(returnVal[i]->to_string()) == curLiveMap.end()) {
            SPDLOG_DEBUG("Return variable {} at line {} is removed", returnVal[i]->to_string(), row_);
            delete returnVal[i];
            returnVal[i] = nullptr; 
            ret.modified = true;
        }
    }

    for (std::size_t i = 0; i < returnVal.size();) {
        if (returnVal[i] == nullptr) {
            returnVal.erase(returnVal.begin() + i);
        } else {
            ++ i;
        }
    }

    if (returnVal.size() == 0) {
        SPDLOG_DEBUG("Return statement at line {} is removed", row_);
        ret.keepStatement = false;
    }

    return ret;
}

void PTO_TUPLE_VAR::remove_variable(const std::vector<int>& removeList) {
    for (const auto& index : removeList) {
        if (index >= (int)varList.size()) {
            SPDLOG_ERROR("Unexpected Error");
            continue;
        }
        SPDLOG_DEBUG("Tuple element {} at line {} is removed", varList[index]->to_string(), row_);
        delete varList[index];
        varList[index] = nullptr;
    }

    for (std::size_t i = 0; i < varList.size();) {
        if (varList[i] == nullptr) {
            varList.erase(varList.begin() + i);
        } else {
            ++ i;
        }
    }
}

void PTO_CALL::remove_variable(const std::vector<int>& removeList) {
    for (const auto& index : removeList) {
        if (index >= (int)arguments.size()) {
            SPDLOG_ERROR("Unexpected Error");
            continue;
        }
        SPDLOG_DEBUG("Argments [{}] at line {} is removed", index, row_);
        delete arguments[index];
        arguments[index] = nullptr;
    }

    for (std::size_t i = 0; i < arguments.size();) {
        if (arguments[i] == nullptr) {
            arguments.erase(arguments.begin() + i);
        } else {
            ++ i;
        }
    }
}

const struct DEAD_CODE_RET PTO_ASSIGNMENT::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    // 不会是yield语句
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.yield_") {
        SPDLOG_ERROR("Should call remove_yield() before calling eliminate_dead_code()");
        return ret;
    }

    // 输入是类SSA格式，左侧可能是tuple
    if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (curLiveMap.find(lhs->to_string()) == curLiveMap.end()) {
            ret.keepStatement = false;
        }
        else if (value->type() == PTO_NODE_TYPE::VARIABLE && lhs->to_string() == value->to_string()) {
            // A = A的无意义赋值
            ret.keepStatement = false;
        }
    }
    else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
        // 右侧必须是用户自定义函数的调用
        if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name().substr(0, 5) == "self.") {
            // 右侧用户函数可能被多次调用，每次调用所需要的输出可能不确定，所以在当前时间点，没法确认左侧哪些输出应当被删除
            // 后续确认了用户函数的输出再处理左侧
        }
        else {
            SPDLOG_ERROR("Unexpected Error");
        }
    }
    else {
        SPDLOG_ERROR("Unexpected Error");
    }

    return ret;
}

const struct DEAD_CODE_RET PTO_FOR_LOOP::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    // 先处理内部的statements
    for (auto& s : statements) {
        auto temp = s->eliminate_dead_code();

        if (!temp.keepStatement) {
            SPDLOG_DEBUG("Statement at line {} is removed", s->row());
            delete s;
            s = nullptr;
            ret.modified = true;
        } else {
            ret.modified |= temp.modified;
        }
    }

    for (std::size_t i = 0; i < statements.size();) {
        if (statements[i] == nullptr) {
            statements.erase(statements.begin() + i);
        } else {
            ++ i;
        }
    }

    if (statements.size() == 0) {
        // For loop 被清空了？
        SPDLOG_DEBUG("For loop at line {} is entirely removed", row_);
        ret.keepStatement = false;
        return ret;
    }

    return ret;
}

const struct DEAD_CODE_RET PTO_IF::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    for (std::size_t i = 0; i < ifStatement.size();) {
        auto temp = ifStatement[i]->eliminate_dead_code();
        if (!temp.keepStatement) {
            SPDLOG_DEBUG("Statemet at line {} is removed", ifStatement[i]->row());
            delete ifStatement[i];
            ifStatement.erase(ifStatement.begin() + i);
            ret.modified = true;
        } else {
            ret.modified |= temp.modified;
            ++ i;
        }
    }

    if (ifStatement.size() == 0) {
        SPDLOG_ERROR("No valid statement in if clause, which is not supported in current dump function");
        return ret;
    }

    for (std::size_t i = 0; i < elseStatement.size();) {
        auto temp = elseStatement[i]->eliminate_dead_code();
        if (!temp.keepStatement) {
            SPDLOG_DEBUG("Statemet at line {} is removed", elseStatement[i]->row());
            delete elseStatement[i];
            elseStatement.erase(elseStatement.begin() + i);
            ret.modified = true;
        } else {
            ret.modified |= temp.modified;
            ++ i;
        }
    }

    if (elseStatement.size() == 0) {
        SPDLOG_ERROR("No valid statement in else clause, which is not supported in current dump function");
        return ret;
    }

    return ret;
}

void PTO_FUNC::adjust_user_func_input() {
    // 扫描内部的statements
    for (auto& s : statements) {
        s->adjust_user_func_input();
    }
}

void PTO_FOR_LOOP::adjust_user_func_input() {
    // 扫描内部的statements
    for (auto& s : statements) {
        s->adjust_user_func_input();
    }
}

void PTO_IF::adjust_user_func_input() {
    comparator->adjust_user_func_input();

    // 扫描内部的statements
    for (auto& s : ifStatement) {
        s->adjust_user_func_input();
    }
    for (auto& s : elseStatement) {
        s->adjust_user_func_input();
    }
}

void PTO_ASSIGNMENT::adjust_user_func_input() {
    // 右侧是用户定义函数？
    if (value->type() != PTO_NODE_TYPE::FUNC_CALL) return;
    auto ptr = (PTO_CALL*)value;

    if (ptr->get_func_name().substr(0, 5) != "self.") return;

    // 是用户自定义函数，先删除多余入参
    if (removedFuncInput.find(ptr->get_func_name().substr(5)) == removedFuncInput.end()) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }
    ptr->remove_variable(removedFuncInput[ptr->get_func_name().substr(5)]);

    // 如果左侧是单一变量，则不处理
    if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (((PTO_VARIABLE*)lhs)->get_type_str().size() != 1) {
            SPDLOG_ERROR("Should call remove_indexed_var() before calling eliminate_dead_code()");
            return;
        }
        return;
    }

    // 左侧应当是tuple variable
    if (lhs->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }
    
    // 删除左侧tuple里多余的输出
    ((PTO_TUPLE_VAR*)lhs)->remove_variable(removedFuncOutput[ptr->get_func_name().substr(5)]);

    // 左侧可能只剩一个变量，转化为普通的variable
    if (((PTO_TUPLE_VAR*)lhs)->get_var_list().size() != 1) {
        return;
    }

    PTO_VARIABLE *newLhs = new PTO_VARIABLE(((PTO_TUPLE_VAR*)lhs)->get_var_list()[0]->to_string(), lhs->row(), lhs->col());
    newLhs->add_type_str(((PTO_VARIABLE*)((PTO_TUPLE_VAR*)lhs)->get_var_list()[0])->get_type_str()[0]);
    delete lhs;
    lhs = newLhs;
}
}