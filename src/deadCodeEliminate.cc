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

    std::unordered_set<int> requiredReturnIndices;
};

typedef std::unordered_map<std::string, CallGraphNode*> STR_GRAPH_MAP;

bool PTO_MODULE::dead_code_eliminate() {
    // 先构建调用图
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

    // 按in_degree倒序处理
    std::vector<CallGraphNode*> processList;
    for (const auto& it : funcInfo) {
        if (it.second->in_degree != 0) continue;
        processList.emplace_back(it.second);
    }
    while (processList.size() != 0) {
        auto ptr = processList.back();
        processList.pop_back();

        // 死代码消除逻辑
        SPDLOG_INFO("Process {}", ptr->func->get_func_name());


        // 更新in_degree
        for (const auto& c : ptr->callees) {
            c->in_degree -= 1;
            if (c->in_degree == 0) {
                processList.emplace_back(c);
            }
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
}
