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
                for (std::size_t i = 0; i < p->func->get_return_type().size(); ++i) {
                    requiredReturnIndices[p->func->get_func_name()].insert(i);
                }
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
                SPDLOG_WARN("Function {} at line {} is not called by any other functions");
                continue;
            }

            const auto ret = p->func->eliminate_dead_code();
            eliminated |= ret.modified;
        }

        // 如果没有死代码被消除则退出循环
        if (!eliminated) break;

        // 有死代码消除的情况下，需要再顺序检查一遍，修改函数调用的入参和返回值


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
    
    // 修改返回值

    // 修改入参

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
        SPDLOG_DEBUG("Variable {} added to live map", varName);
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
                if (curLiveMap.find(forInitVar.back()[i]) == curLiveMap.end()) continue;

                ret |= list[i]->add_to_live_map();
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
        SPDLOG_DEBUG("Variable {}[{}] added to live map", varName, index[0]);
        return true;
    }
    else {
        // 有可能是不同的index
        auto& ptr = curLiveMap[varName];

        if (ptr.partLive && ptr.index.find(index[0]) == ptr.index.end()) {
            ptr.index.insert(index[0]);

            SPDLOG_DEBUG("Variable {}[{}] added to live map", varName, index[0]);

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

const struct DEAD_CODE_RET PTO_ASSIGNMENT::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;
    // if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
    //     if (liveMap.find(lhs->to_string()) == liveMap.end()) {
    //         ret.keepStatement = false;
    //     }
    // }
    // else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE || lhs->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
    //     std::vector<PTO_BASE*> varList;
    //     if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
    //         varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
    //     } else {
    //         varList = ((PTO_LIST_VAR*)lhs)->get_var_list();
    //     }

    //     ret.keepStatement = false;
    //     for (const auto& arg : varList) {
    //         if (arg->type() != PTO_NODE_TYPE::VARIABLE) {
    //             SPDLOG_ERROR("Unexpected Error");
    //         } else {
    //             ret.keepStatement |= (liveMap.find(arg->to_string()) != liveMap.end());
    //         }
    //     }
    // }
    // else {
    //     SPDLOG_ERROR("Unexpected Error");
    // }

    return ret;
}

const struct DEAD_CODE_RET PTO_FOR_LOOP::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;

    // // 先确认内部的statements是否需要
    // ret.modified = false;
    // for (std::size_t i = 0; i < statements.size() - 1;) {
    //     auto temp = statements[i]->eliminate_dead_code(liveMap);
        
    //     if (!temp.keepStatement) {
    //         SPDLOG_DEBUG("Statements at line {} is deleted", statements[i]->row());
    //         delete statements[i];
    //         statements.erase(statements.begin() + i);
    //         ret.modified = true;
    //     } else {
    //         ret.modified |= temp.modified;
    //         ++ i;
    //     }
    // }

    // if (statements.size() == 1) {
    //     // For loop被清空了？
    //     SPDLOG_WARN("For loop at line {} has no useful statement", row_);
    //     ret.keepStatement = false;
    //     return ret;
    // }

    // // 同步处理initVar和info里的init_values
    // // 记录哪些index的variable需要删掉
    // std::vector<int> deleteList;
    // for (std::size_t i = 0; i < initVar.size(); ++i) {
    //     if (liveMap.find(initVar[i]->to_string()) == liveMap.end()) {
    //         deleteList.emplace_back(i);
    //     }
    // }
    // // 更新initVar
    // for (std::size_t i = 0; i < initVar.size();) {
    //     if (liveMap.find(initVar[i]->to_string()) == liveMap.end()) {
    //         ret.modified = true;
    //         delete initVar[i];
    //         initVar.erase(initVar.begin() + i);
    //     } else {
    //         ++ i;
    //     }
    // }

    // // 处理info里的init_values
    // const auto& argList = info->get_arguments();
    // for (auto& arg : argList) {
    //     if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "init_values") {
    //         auto ptr = (PTO_KEYWORD*)arg;
    //         if (ptr->get_value()->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
    //             SPDLOG_ERROR("Unexpected Error");
    //         }

    //         ((PTO_TUPLE_VAR*)ptr->get_value())->delete_dead_code(deleteList);
    //     }
    // }
    
    // // 处理yield语句
    // PTO_ASSIGNMENT *assignPtr = (PTO_ASSIGNMENT*)statements.back();
    // if (assignPtr->get_lhs()->type() == PTO_NODE_TYPE::VARIABLE || assignPtr->get_lhs()->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
    //     if (deleteList.size() != 0) {
    //         // 删掉唯一的variable??
    //         SPDLOG_ERROR("Unexpected Error");
    //     }
    // }
    // else if (assignPtr->get_lhs()->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
    //     ((PTO_TUPLE_VAR*)assignPtr->get_lhs())->delete_dead_code(deleteList);
    // }
    // else {
    //     SPDLOG_ERROR("Unexpected Error");
    // }

    // // 处理yield的入参
    // PTO_CALL *yieldPtr = (PTO_CALL*)assignPtr->get_value();
    // yieldPtr->delete_dead_code(deleteList);

    return ret;
}

const struct DEAD_CODE_RET PTO_IF::eliminate_dead_code() {
    struct DEAD_CODE_RET ret;
    
    // for (std::size_t i = 0; i < ifStatement.size();) {
    //     auto temp = ifStatement[i]->eliminate_dead_code(liveMap);
    //     if (!temp.keepStatement) {
    //         SPDLOG_DEBUG("Statement at line {} is deleted", ifStatement[i]->row());
    //         delete ifStatement[i];
    //         ifStatement.erase(ifStatement.begin() + i);
    //     } else {
    //         ret.modified |= temp.modified;
    //         ++ i;
    //     }
    // }

    // if (ifStatement.size() == 0) {
    //     SPDLOG_ERROR("No valid if statement at line {}", row_);
    // }

    // for (std::size_t i = 0; i < elseStatement.size();) {
    //     auto temp = elseStatement[i]->eliminate_dead_code(liveMap);
    //     if (!temp.keepStatement) {
    //         SPDLOG_DEBUG("Statement at line {} is deleted", elseStatement[i]->row());
    //         delete elseStatement[i];
    //         elseStatement.erase(elseStatement.begin() + i);
    //     } else {
    //         ret.modified |= temp.modified;
    //         ++ i;
    //     }
    // }

    // if (elseStatement.size() == 0) {
    //     SPDLOG_ERROR("No valid else statement at line {}", row_);
    // }
    

    return ret;
}


// void PTO_TUPLE_VAR::delete_dead_code() {
//     for (const auto& i : deleteList) {
//         if ((int)varList.size() <= i) {
//             SPDLOG_ERROR("Unexpected Error");
//             return;
//         }
//         delete varList[i];
//         varList[i] = nullptr;
//     }
//     for (std::size_t i = 0; i < varList.size();) {
//         if (varList[i] == nullptr) {
//             varList.erase(varList.begin() + i);
//         } else {
//             ++i;
//         }
//     }
// }

// void PTO_CALL::delete_dead_code(const std::vector<int>& deleteList) {
//     // 只适用于yield_
//     if (funcName != "pypto.language.yield_") {
//         SPDLOG_ERROR("Unexpected error");
//         return;
//     }

//     for (const auto& i : deleteList) {
//         if ((int)arguments.size() <= i) {
//             SPDLOG_ERROR("Unexpected Error");
//             return;
//         }
//         delete arguments[i];
//         arguments[i] = nullptr;
//     }
//     for (std::size_t i = 0; i < arguments.size();) {
//         if (arguments[i] == nullptr) {
//             arguments.erase(arguments.begin() + i);
//         } else {
//             ++i;
//         }
//     }
// }

// void PTO_RETURN::delete_dead_code(const std::unordered_set<int>& requiredReturn) {
//     for (std::size_t i = 0; i < returnVal.size(); ++i) {
//         if (requiredReturn.find(i) == requiredReturn.end()) {
//             delete returnVal[i];
//             returnVal[i] = nullptr;
//         }
//     }
//     for (std::size_t i = 0; i < returnVal.size();) {
//         if (returnVal[i] == nullptr) {
//             returnVal.erase(returnVal.begin() + i);
//         } else {
//             ++i;
//         }
//     }
// }


// void PTO_IF::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     comparator->collect_required_return(liveMap);
    
//     for (const auto& s : ifStatement) {
//         s->collect_required_return(liveMap);
//     }
//     for (const auto& s : elseStatement) {
//         s->collect_required_return(liveMap);
//     }
// }

// void PTO_FOR_LOOP::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     for (const auto& s : statements) {
//         s->collect_required_return(liveMap);
//     }
// }

// void PTO_RETURN::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     for (const auto& s : returnVal) {
//         s->collect_required_return(liveMap);
//     }
// }

// void PTO_ASSIGNMENT::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     // 右侧是pto_call?
//     if (value->type() != PTO_NODE_TYPE::FUNC_CALL) return;
//     const auto& funcName = ((PTO_CALL*)value)->get_func_name();
//     if (funcName.substr(0, 5) != "self.") return;

//     if (funcRequiredReturn.find(funcName.substr(5)) == funcRequiredReturn.end()) {
//         SPDLOG_ERROR("Unexpected Error");
//         return;
//     }

//     // 强制要求左值是variable
//     if (lhs->type() != PTO_NODE_TYPE::TYPED_VARIABLE) {
//         SPDLOG_ERROR("Unexpected ERROR");
//         return;
//     }

//     if (liveMap.find(lhs->to_string()) == liveMap.end()) {
//         // 不存在即为dead code，则不可能调用到这
//         SPDLOG_ERROR("Unexpected Error");
//         return;
//     }

//     const auto& status = liveMap.find(lhs->to_string())->second;

//     if (status.partLive) {
//         // 只有部分结果被使用
//         for (const auto& it : status.index) {
//             funcRequiredReturn[funcName.substr(5)].insert(it);
//         }
//     } else {
//         // 全部被使用
//         // 根据typestr的数量决定
//         for (std::size_t i = 0; i < ((PTO_VARIABLE*)lhs)->get_type_str().size(); ++i) {
//             funcRequiredReturn[funcName.substr(5)].insert(i);
//         }
//     }
        
// }

// void PTO_KEYWORD::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     value->collect_required_return(liveMap);
// }

// // void PTO_CALL::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
// //     // 只记录self.xxx函数
// //     if (funcName.substr(0, 5) == "self.") {
// //         if (funcRequiredReturn.find(funcName.substr(5)) == funcRequiredReturn.end()) {
// //             SPDLOG_ERROR("Unexpected Error");
// //         }
// //         callees.insert(funcName.substr(5));
// //     }
// // }

// void PTO_BINARY_OP::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     lhs->collect_required_return(liveMap);
//     rhs->collect_required_return(liveMap);
// }

// void PTO_LIST_VAR::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     for (const auto& s : varList) {
//         s->collect_required_return(liveMap);
//     }
// }

// void PTO_TUPLE_VAR::collect_required_return(std::unordered_map<std::string, struct LIVE_VARIABLE>& liveMap) const {
//     for (const auto& s : varList) {
//         s->collect_required_return(liveMap);
//     }
// }


}