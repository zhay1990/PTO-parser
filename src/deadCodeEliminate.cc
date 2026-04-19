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

struct LIVE_VARIABLE {
    bool partLive;
    std::string varName;
    std::unordered_set<int> index;

    LIVE_VARIABLE()
        : partLive(false),
          varName(),
          index()
    {}

    LIVE_VARIABLE(const std::string& n)
        : partLive(false),
          varName(n),
          index()
    {}

    LIVE_VARIABLE(const std::string& n, const int& i)
        : partLive(true),
          varName(n),
          index()
    {index.insert(i);}
};

// 记录所有函数哪些输出是需要的
static std::unordered_map<std::string, std::unordered_set<int>> requiredReturnIndices;
// 记录所有函数哪些输入被删除
static std::unordered_map<std::string, std::vector<int>> removedFuncInput;
// 记录所有函数哪些输出被删除
static std::unordered_map<std::string, std::vector<int>> removedFuncOutput;

// 记录当前函数的哪些输出是需要的，在处理return statement时使用
static std::unordered_set<int> curRequiredReturnIndices;
// 记录当前函数的有效变量
static std::unordered_map<std::string, struct LIVE_VARIABLE> curLiveMap;
// 记录for loop的init var，用于处理yield语句
static std::vector<std::vector<std::string>> forInitVar;
// 记录每个用户定义函数的返回值被赋值给了哪些变量，和removedFuncOutput配合完成变量类型和潜在的indexed variable的序号修改
static std::unordered_map<std::string, std::unordered_set<std::string>> funcReturnAssignVar;


typedef std::unordered_map<std::string, CallGraphNode*> STR_GRAPH_MAP;
typedef std::unordered_map<std::string, struct LIVE_VARIABLE> STR_LIVE_MAP;

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
            funcReturnAssignVar.clear();
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
    SPDLOG_DEBUG("Processing func {}", funcName);

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
    // 先记录这个for循环的init var名字，在后续处理yield语句时需要
    forInitVar.emplace_back(std::vector<std::string>());

    for (const auto& v : initVar) {
        forInitVar.back().emplace_back(v->to_string());
    }
    
    // 循环处理，确定所有的live variable
    bool ret = false;
    while (true) {
        bool varAdded = false;

        // 倒序处理所有语句
        for (int i = (int)statements.size() - 1; i >= 0; --i) {
            varAdded |= statements[i]->add_to_live_map();
        }

        if (!varAdded) break;

        ret = true;
    }

    // 处理info里的init values
    for (const auto& arg : info->get_arguments()) {
        if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "init_values") {
            const auto& tuple = ((PTO_KEYWORD*)arg)->get_value();
            if (tuple->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return ret;
            }

            const auto& varList = ((PTO_TUPLE_VAR*)tuple)->get_var_list();
            if (varList.size() != initVar.size()) {
                SPDLOG_ERROR("Unexpected Error");
                return ret;
            }

            for (std::size_t i = 0; i < varList.size(); ++i) {
                if (varList[i]->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_ERROR("Unexpected Error");
                    return ret;
                }

                if (curLiveMap.find(initVar[i]->to_string()) != curLiveMap.end()) {
                    ret |= varList[i]->add_to_live_map();
                }
            }
        }
    }

    forInitVar.pop_back();

    return ret;
}

bool PTO_VARIABLE::add_to_live_map() const {
    if (curLiveMap.find(varName) == curLiveMap.end()) {
        curLiveMap[varName] = LIVE_VARIABLE(varName);
        SPDLOG_DEBUG("Variable {} at line {} added to live map", varName, row_);
        return true;
    }
    return false;
}

bool PTO_ASSIGNMENT::add_to_live_map() const {
    // 对于yield的赋值语句需要特殊处理
    bool ret = false;
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.yield_") {
        if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            if (forInitVar.size() == 0 || forInitVar.back().size() != 1) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }

            // 如果initVar在liveMap里，则lhs也应当被加入
            if (curLiveMap.find(forInitVar.back()[0]) != curLiveMap.end()) {
                ret |= lhs->add_to_live_map();
            }
            // initVar和yield语句的参数应当保持一致，所以如果initVar不在liveMap里，但lhs在，则initVar也当被加入
            else if (curLiveMap.find(lhs->to_string()) != curLiveMap.end()) {
                ret = true;
                curLiveMap[forInitVar.back()[0]] = LIVE_VARIABLE(forInitVar.back()[0]);
                SPDLOG_DEBUG("Variable {} at line {} added to live map", forInitVar.back()[0], row_);
            }

            // 处理yield的入参
            if (curLiveMap.find(lhs->to_string()) != curLiveMap.end()) {
                const auto& argList = ((PTO_CALL*)value)->get_arguments();

                if (argList.size() != 1) {
                    SPDLOG_ERROR("Unexpected Error");
                    return false;
                }

                ret |= argList[0]->add_to_live_map();
            }
        }
        else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE || lhs->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            std::vector<PTO_BASE*> list;

            if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
                list = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            } else {
                list = ((PTO_LIST_VAR*)lhs)->get_var_list();
            }

            if (forInitVar.size() == 0 || forInitVar.back().size() != list.size()) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }

            for (std::size_t i = 0; i < forInitVar.back().size(); ++i) {
                if (list[i]->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_ERROR("Unexpected Error");
                    return false;
                }

                if (curLiveMap.find(forInitVar.back()[i]) != curLiveMap.end()) {
                    ret |= list[i]->add_to_live_map();
                }
                else if (curLiveMap.find(list[i]->to_string()) != curLiveMap.end()) {
                    ret = true;
                    curLiveMap[forInitVar.back()[i]] = LIVE_VARIABLE(forInitVar.back()[i]);
                    SPDLOG_DEBUG("Variable {} at line {} added to live map", forInitVar.back()[i], row_);
                }
            }

            // 处理yield的入参
            const auto& argList = ((PTO_CALL*)value)->get_arguments();
            if (argList.size() != list.size()) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }

            for (std::size_t i = 0; i < list.size(); ++i) {
                if (list[i]->type() != PTO_NODE_TYPE::VARIABLE || argList[i]->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_ERROR("Unexpected Error");
                    return false;
                }
                if (curLiveMap.find(list[i]->to_string()) != curLiveMap.end()) {
                    ret |= argList[i]->add_to_live_map();
                }
            }
        }
        else {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }
    }
    else {
        // 处理的是SSA，所以左值只会是variable
        if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            if (curLiveMap.find(lhs->to_string()) != curLiveMap.end()) {
                // 左值被需要
                ret |= value->add_to_live_map();

                // 右值是用户自定义函数？
                if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name().substr(0, 5) == "self.") {
                    const auto& info = curLiveMap.find(lhs->to_string())->second;

                    std::string userFuncName = ((PTO_CALL*)value)->get_func_name().substr(5);
                    if (!info.partLive) {
                        // 这个函数的所有输出都需要，插入-1表示该信息
                        requiredReturnIndices[userFuncName].insert(-1);
                        SPDLOG_DEBUG("All outputs of function {} are required from line {}", userFuncName, row_);
                    }
                    else {
                        requiredReturnIndices[userFuncName].insert(info.index.begin(), info.index.end());
                        for (const auto& it : info.index) {
                            SPDLOG_DEBUG("The [{}] outputs of function {} are required from line {}", it, userFuncName, row_);
                        }
                    }
                }
            }
        }
        else {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }
    }

    return ret;
}

bool PTO_INDEXED_VAR::add_to_live_map() const {
    if (curLiveMap.find(varName) == curLiveMap.end()) {
        curLiveMap[varName] = LIVE_VARIABLE(varName, index[0]);
        SPDLOG_DEBUG("Variable {}[{}] at line {} added to live map", varName, index[0], row_);
        return true;
    }
    else {
        // 有可能是不同的index
        auto& ptr = curLiveMap[varName];

        if (ptr.partLive && ptr.index.find(index[0]) == ptr.index.end()) {
            ptr.index.insert(index[0]);

            SPDLOG_DEBUG("Variable {}[{}] at line added to live map", varName, index[0], row_);

            return true;
        }
    }

    return false;
}

bool PTO_CALL::add_to_live_map() const {
    // yield函数需要单独处理，不在这处理
    if (funcName == "pypto.language.yield_") {
        SPDLOG_ERROR("Unexpected Error at line {}", row_);
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

void PTO_LIST_VAR::remove_variable(const std::vector<int>& removeList) {
    for (const auto& index : removeList) {
        if (index >= (int)varList.size()) {
            SPDLOG_ERROR("Unexpected Error");
            continue;
        }
        SPDLOG_DEBUG("LIST element {} at line {} is removed", varList[index]->to_string(), row_);
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

void PTO_KEYWORD::remove_variable(const std::vector<int>& removeList) {
    switch (value->type()) {
        case PTO_NODE_TYPE::TUPLE_VARIABLE: 
            ((PTO_TUPLE_VAR*)value)->remove_variable(removeList);
            break;
        case PTO_NODE_TYPE::LIST_VARIABLE:
            ((PTO_LIST_VAR*)value)->remove_variable(removeList);
            break;
        default:
            SPDLOG_ERROR("Unexpected call to PTO_KEYWORD::remove_variable");
            break;
    }
}

const struct DEAD_CODE_RET PTO_ASSIGNMENT::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    // 如果是yield语句则需要特殊处理
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.yield_") {
        if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            if (curLiveMap.find(lhs->to_string()) == curLiveMap.end()) {
                ret.keepStatement = false;
            }
        }
        else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE || lhs->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            // 每个变量单独分析
            std::vector<PTO_BASE*> varList;
            if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
                varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            } else {
                varList = ((PTO_LIST_VAR*)lhs)->get_var_list();
            }

            std::vector<int> removeList;
            for (std::size_t i = 0; i < varList.size(); ++i) {
                if (varList[i]->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_ERROR("Unexpected Error");
                    return ret;
                }

                if (curLiveMap.find(varList[i]->to_string()) == curLiveMap.end()) {
                    removeList.emplace_back(i);
                }
            }

            if (removeList.size() == varList.size()) {
                ret.keepStatement = false;
                return ret;
            }

            if (removeList.size() == 0) {
                return ret;
            }
            
            ret.modified = true;

            if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
                ((PTO_TUPLE_VAR*)lhs)->remove_variable(removeList);
            }
            else {
                ((PTO_LIST_VAR*)lhs)->remove_variable(removeList);
            }
            
            ((PTO_CALL*)value)->remove_variable(removeList);
        }
        else {
            SPDLOG_ERROR("Unexpected Error");
            return ret;
        }
    }
    // 只支持SSA格式，所以左侧是单变量
    else if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (curLiveMap.find(lhs->to_string()) == curLiveMap.end()) {
            ret.keepStatement = false;
        }
    }
    else {
        SPDLOG_ERROR("Non-SSA assignment at line {}", row_);
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
    
    // 处理initVar
    std::vector<int> removeList;
    for (std::size_t i = 0; i < initVar.size(); ++i) {
        if (curLiveMap.find(initVar[i]->to_string()) == curLiveMap.end()) {
            SPDLOG_DEBUG("For loop init variable {} at line {} is removed", initVar[i]->to_string(), row_);
            delete initVar[i];
            initVar[i] = nullptr;
            removeList.emplace_back(i);
            ret.modified = true;
        }
    }
    for (std::size_t i = 0; i < initVar.size();) {
        if (initVar[i] == nullptr) {
            initVar.erase(initVar.begin() + i);
        } else {
            ++ i;
        }
    }

    const auto& argList = info->get_arguments();
    for (auto& arg : argList) {
        if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "init_values") {
            ((PTO_KEYWORD*)arg)->remove_variable(removeList);
        }
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
    // 先处理一个问题，右侧的表达式中是否有和用户函数返回值相关联的变量，可能需要调整index，甚至从tuple variable转成普通variable
    value->adjust_user_func_input();

    // 如果右侧是index var，且对应的函数只有一个output，则上面函数会将其index清空，这里识别下，将其替换为普通variable
    if (value->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)value)->get_index().size() == 0) {
        auto newValue = new PTO_VARIABLE(((PTO_INDEXED_VAR*)value)->get_var_name(), value->row(), value->col());
        SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newValue->to_string(), value->row());
        delete value;
        value = newValue;
    }



    // 会出现传递的情况吗？即
    // varA = self.xxx()
    // varB = varA
    // xxx = varB[1]
    // 这种情况？
    if (value->type() == PTO_NODE_TYPE::VARIABLE) {
        for (auto& it : funcReturnAssignVar) {
            if (it.second.find(value->to_string()) != it.second.end()) {
                SPDLOG_DEBUG("Variable {} is the output of func {} and is propogate to variable {} at line {}",
                    value->to_string(),
                    it.first,
                    lhs->to_string(),
                    row_
                );
                it.second.insert(lhs->to_string());
            }
        }
    }

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

    if (lhs->type() != PTO_NODE_TYPE::TYPED_VARIABLE) {
        SPDLOG_ERROR("Non-SSA format assignment at line {}", row_);
        return;
    }
    // 记录左侧变量为该函数的输出
    // 该变量是否被覆盖了？
    for (auto& it : funcReturnAssignVar) {
        if (it.second.find(lhs->to_string()) != it.second.end()) {
            SPDLOG_DEBUG("Variable {} is changed from func {} to func {} at line {}",
                lhs->to_string(),
                it.first,
                ptr->get_func_name(),
                row_
            );
            it.second.erase(lhs->to_string());
        }
    }
    funcReturnAssignVar[ptr->get_func_name().substr(5)].insert(lhs->to_string());

    // 调整左侧的变量类型
    lhs->adjust_user_func_input();
}

void PTO_VARIABLE::adjust_user_func_input() {
    // 普通变量不需要调整
    if (typeStr.size() == 0) return;

    // 先确定这个变量是哪个函数的
    for (const auto& it : funcReturnAssignVar) {
        if (it.second.find(varName) == it.second.end()) continue;

        const auto& removeList = removedFuncOutput[it.first];

        for (const auto& index : removeList) {
            if (index >= (int)typeStr.size()) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            SPDLOG_DEBUG("The type {} is removed from variable {} at line {}", typeStr[index], varName, row_);
            typeStr[index] = "";
        }
        for (std::size_t i = 0; i < typeStr.size();) {
            if (typeStr[i] == "") {
                typeStr.erase(typeStr.begin() + i);
            } else {
                ++ i;
            }
        }
    }
}

void PTO_TUPLE_VAR::adjust_user_func_input() {
    for (auto& var : varList) {
        var->adjust_user_func_input();

        // var有可能是indexed variable，并且index被清空
        if (var->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)var)->get_index().size() == 0) {
            auto newVar = new PTO_VARIABLE(((PTO_INDEXED_VAR*)var)->get_var_name(), var->row(), var->col());
            SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newVar->to_string(), var->row());
            delete var;
            var = newVar;
        }
    }
}

void PTO_LIST_VAR::adjust_user_func_input() {
    for (auto& var : varList) {
        var->adjust_user_func_input();

        // var有可能是indexed variable，并且index被清空
        if (var->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)var)->get_index().size() == 0) {
            auto newVar = new PTO_VARIABLE(((PTO_INDEXED_VAR*)var)->get_var_name(), var->row(), var->col());
            SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newVar->to_string(), var->row());
            delete var;
            var = newVar;
        }
    }
}

void PTO_BINARY_OP::adjust_user_func_input() {
    lhs->adjust_user_func_input();
    rhs->adjust_user_func_input();

    // 有可能lhs是indexed var，并被改造成普通variable
    if (lhs->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)lhs)->get_index().size() == 0) {
        auto newlhs = new PTO_VARIABLE(((PTO_INDEXED_VAR*)lhs)->get_var_name(), lhs->row(), lhs->col());
        SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newlhs->to_string(), lhs->row());
        delete lhs;
        lhs = newlhs;
    }

    // 有可能rhs是indexed var，并被改造成普通variable
    if (rhs->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)rhs)->get_index().size() == 0) {
        auto newrhs = new PTO_VARIABLE(((PTO_INDEXED_VAR*)rhs)->get_var_name(), rhs->row(), rhs->col());
        SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newrhs->to_string(), rhs->row());
        delete rhs;
        lhs = newrhs;
    }
}

void PTO_INDEXED_VAR::adjust_user_func_input() {
    // 先确认该变量属于哪个user function的输出
    for (const auto& it : funcReturnAssignVar) {
        if (it.second.find(varName) == it.second.end()) continue;

        // 拿到该函数的删除列表
        const auto& removeList = removedFuncOutput[it.first];

        // 当前的index不应当在这个列表里
        // 每有一个比当前index小的remove index，则当前index需要-1
        int reduceCount = 0;
        for (const auto& r : removeList) {
            if (r == index[0]) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }
            else if (r < index[0]) {
                reduceCount += 1;
            }
        }
        
        index[0] -= reduceCount;

        // 特殊处理，当前变量类型从tuple变回普通变量，删除index
        if (requiredReturnIndices[it.first].size() == 1) {
            if (index[0] != 0) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            index.clear();
        }
    }
}

void PTO_CALL::adjust_user_func_input() {
    for (auto& arg : arguments) {
        arg->adjust_user_func_input();

        // 特殊处理
        if (arg->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)arg)->get_index().size() == 0) {
            auto newArg = new PTO_VARIABLE(((PTO_INDEXED_VAR*)arg)->get_var_name(), arg->row(), arg->col());
            SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newArg->to_string(), arg->row());
            delete arg;
            arg = newArg;
        }
    }
}

void PTO_KEYWORD::adjust_user_func_input() {
    value->adjust_user_func_input();

    // 特殊处理
    if (value->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)value)->get_index().size() == 0) {
        auto newValue = new PTO_VARIABLE(((PTO_INDEXED_VAR*)value)->get_var_name(), value->row(), value->col());
        SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newValue->to_string(), value->row());
        delete value;
        value = newValue;
    }
}

void PTO_RETURN::adjust_user_func_input() {
    for (auto& r : returnVal) {
        r->adjust_user_func_input();

        // 特殊处理
        if (r->type() == PTO_NODE_TYPE::INDEXED_VARIABLE && ((PTO_INDEXED_VAR*)r)->get_index().size() == 0) {
            auto newR = new PTO_VARIABLE(((PTO_INDEXED_VAR*)r)->get_var_name(), r->row(), r->col());
            SPDLOG_DEBUG("Indexed variable {} is changed to normal variable at line {}", newR->to_string(), r->row());
            delete r;
            r = newR;
        }

    }
}

}