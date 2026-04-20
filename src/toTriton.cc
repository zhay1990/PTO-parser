#include "ptoNode.hh"
#include "logger.hh"

namespace pto_parser
{

// 确认哪些input arg是program id
static std::unordered_map<std::string, std::unordered_set<std::string>> programIdArg;

void PTO_MODULE::convert_to_triton(const std::string& fileName) const {
    // 简化处理，假设一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return;
    }

    ((PTO_CLASS*)classOrFunc[0])->convert_to_triton(fileName);
}

void PTO_CLASS::convert_to_triton(const std::string& fileName) const {
    // 寻找到host函数，现在假定只有一个host函数
    PTO_FUNC *ptr = nullptr;

    for (const auto& func : functions) {
        if (func->get_decorate() == "pypto.language.function(type = pypto.language.FunctionType.Orchestration)") {
            if (ptr != nullptr) {
                SPDLOG_ERROR("More than one host function is founded, which is not supported");
                return;
            }
            ptr = func;
        }
    }

    SPDLOG_INFO("Found host function: {}", ptr->get_func_name());

    // 扫描host函数，确认每个triton kernel函数的哪个入参对应program id
    // 当前要求每个kernel函数的调用必须是parallel循环里的唯一statement
    // 如果不是唯一的，则会把这个parallel循环当成普通的for loop处理
    ptr->determine_program_id_for_triton_kernel();

    for (const auto& it : programIdArg) {
        SPDLOG_DEBUG("For function {}, the program id variable are:", it.first);
        for (const auto& iit : it.second) {
            SPDLOG_DEBUG(iit);
        }
    }

    // 下面先将kernel函数生成出来
}

void PTO_FUNC::determine_program_id_for_triton_kernel() const {
    for (const auto& ptr : statements) {
        ptr->determine_program_id_for_triton_kernel();
    }
}

void PTO_IF::determine_program_id_for_triton_kernel() const {
    for (const auto& ptr : ifStatement) {
        ptr->determine_program_id_for_triton_kernel();
    }
    for (const auto& ptr : elseStatement) {
        ptr->determine_program_id_for_triton_kernel();
    }
}

void PTO_FOR_LOOP::determine_program_id_for_triton_kernel() const {
    for (const auto& ptr : statements) {
        ptr->determine_program_id_for_triton_kernel();
    }

    if (statements.size() != 1) {
        return;
    }

    if (statements[0]->type() != PTO_NODE_TYPE::ASSIGNMENT) {
        return;
    }

    auto ptr = (PTO_ASSIGNMENT*)statements[0];

    if (ptr->get_value()->type() != PTO_NODE_TYPE::FUNC_CALL) {
        return;
    }

    auto funcPtr = (PTO_CALL*)ptr->get_value();

    if (funcPtr->get_func_name().substr(0, 5) != "self.") {
        // 这里依旧假定其他函数都是kernel函数
        return;
    }

    // 循环变量是program id
    programIdArg[funcPtr->get_func_name().substr(5)].insert(iter->to_string());
}
    
} // namespace pto_parser
