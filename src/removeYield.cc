#include "ptoNode.hh"
#include "logger.hh"

namespace pto_parser {

static std::unordered_map<std::string, std::vector<std::string>> VAR_TYPE;
static std::vector<std::vector<std::string>> forLoopInitVarName;

// 辅助函数
bool gen_assign_for_for_loop_init(PTO_FOR_LOOP* forPtr, std::vector<PTO_BASE*>& newStatements) {
    // 将for loop的init value转化为assignment
    auto infoPtr = forPtr->get_info();
    PTO_TUPLE_VAR *initVal = nullptr;
    const auto& initVar = forPtr->get_init_var();

    // 记录init var名字，后续在处理yield语句时需要使用
    forLoopInitVarName.emplace_back(std::vector<std::string>());
    for (const auto& i : initVar) {
        forLoopInitVarName.back().emplace_back(i->to_string());
    }

    for (const auto& arg : infoPtr->get_arguments()) {
        if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "init_values") {
            if (((PTO_KEYWORD*)arg)->get_value()->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }
            initVal = (PTO_TUPLE_VAR*)((PTO_KEYWORD*)arg)->get_value();
        }
    }

    if (initVal == nullptr) {
        SPDLOG_ERROR("Unexpected Error");
        return false;
    }

    for (std::size_t i = 0; i < initVar.size(); ++i) {
        auto lhs = new PTO_VARIABLE(initVar[i]->to_string(), initVar[i]->row(), initVar[i]->col());

        // 强制要求initVal是variable类型
        if (initVal->get_var_list()[i]->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }

        if (VAR_TYPE.find(initVal->get_var_list()[i]->to_string()) == VAR_TYPE.end()) {
            SPDLOG_ERROR("Unexpected Error for var name {} at line {}", initVal->get_var_list()[i]->to_string(), forPtr->row());
            return false;
        }

        for (const auto& t : VAR_TYPE[initVal->get_var_list()[i]->to_string()]) {
            lhs->add_type_str(t);
        }

        VAR_TYPE[lhs->to_string()] = lhs->get_type_str();

        auto assignPtr = new PTO_ASSIGNMENT(lhs, lhs->row(), lhs->col());

        assignPtr->set_value(new PTO_VARIABLE(
            initVal->get_var_list()[i]->to_string(),
            initVal->get_var_list()[i]->row(),
            initVal->get_var_list()[i]->col()
        ));

        newStatements.emplace_back(assignPtr);
    }

    // 删掉for loop里的initVar 和init values
    forPtr->remove_init_var();
    initVal->clear();

    return true;
}

bool convert_yield_to_assign(PTO_ASSIGNMENT* assignPtr, std::vector<PTO_BASE*>& newStatements) {
    if (assignPtr->get_value()->type() != PTO_NODE_TYPE::FUNC_CALL) {
        return false;
    }
    
    auto funcPtr = (PTO_CALL*)assignPtr->get_value();

    if (funcPtr->get_func_name() != "pypto.language.yield_") {
        return false;
    }

    const auto& args = funcPtr->get_arguments();
    // 生成对应的assignment语句
    if (assignPtr->get_lhs()->type() == PTO_NODE_TYPE::TYPED_VARIABLE || assignPtr->get_lhs()->type() == PTO_NODE_TYPE::VARIABLE) {
        if (args.size() != 1 || args[0]->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }

        auto newLhs0 = new PTO_VARIABLE(assignPtr->get_lhs()->to_string(), assignPtr->get_lhs()->row(), assignPtr->get_lhs()->col());

        if (VAR_TYPE.find(args[0]->to_string()) == VAR_TYPE.end()) {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }

        for (const auto& t : VAR_TYPE[args[0]->to_string()]) {
            newLhs0->add_type_str(t);
        }

        VAR_TYPE[newLhs0->to_string()] = newLhs0->get_type_str();

        auto newAssign0 = new PTO_ASSIGNMENT(newLhs0, assignPtr->row(), assignPtr->col());
        newAssign0->set_value(new PTO_VARIABLE(args[0]->to_string(), args[0]->row(), args[0]->col()));

        newStatements.emplace_back(newAssign0);

        auto newLhs1 = new PTO_VARIABLE(forLoopInitVarName.back()[0], assignPtr->row(), assignPtr->col());
        for (const auto& t : VAR_TYPE[args[0]->to_string()]) {
            newLhs1->add_type_str(t);
        }
        VAR_TYPE[newLhs1->to_string()] = newLhs1->get_type_str();

        auto newAssign1 = new PTO_ASSIGNMENT(newLhs1, assignPtr->row(), assignPtr->col());
        newAssign1->set_value(new PTO_VARIABLE(newLhs0->to_string(), assignPtr->row(), assignPtr->col()));

        newStatements.emplace_back(newAssign1);
    }
    else if (assignPtr->get_lhs()->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
        const auto& varList = ((PTO_TUPLE_VAR*)assignPtr->get_lhs())->get_var_list();

        for (std::size_t i = 0; i < varList.size(); ++i) {
            auto newLhs0 = new PTO_VARIABLE(varList[i]->to_string(), varList[i]->row(), varList[i]->col());

            if (VAR_TYPE.find(args[i]->to_string()) == VAR_TYPE.end()) {
                SPDLOG_ERROR("Unexpected Error for var {} at line {}", args[i]->to_string(), assignPtr->row());
                return false;
            }

            for (const auto& t : VAR_TYPE[args[i]->to_string()]) {
                newLhs0->add_type_str(t);
            }

            VAR_TYPE[newLhs0->to_string()] = newLhs0->get_type_str();

            auto newAssign0 = new PTO_ASSIGNMENT(newLhs0, assignPtr->row(), assignPtr->col());
            newAssign0->set_value(new PTO_VARIABLE(args[i]->to_string(), args[i]->row(), args[i]->col()));

            newStatements.emplace_back(newAssign0);

            auto newLhs1 = new PTO_VARIABLE(forLoopInitVarName.back()[i], assignPtr->row(), assignPtr->col());
            for (const auto& t : VAR_TYPE[args[i]->to_string()]) {
                newLhs1->add_type_str(t);
            }
            VAR_TYPE[newLhs1->to_string()] = newLhs1->get_type_str();

            auto newAssign1 = new PTO_ASSIGNMENT(newLhs1, assignPtr->row(), assignPtr->col());
            newAssign1->set_value(new PTO_VARIABLE(newLhs0->to_string(), assignPtr->row(), assignPtr->col()));

            newStatements.emplace_back(newAssign1);
        }
    }
    else {
        SPDLOG_ERROR("Unexpected Error {} at line {}", (int)assignPtr->get_lhs()->type(), assignPtr->row());
        return false;
    }
    return true;
}

bool PTO_MODULE::remove_yield() const {
    // 简化处理，假定一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return false;
    }

    return classOrFunc[0]->remove_yield();
}

bool PTO_CLASS::remove_yield() {
    for (auto& func : functions) {
        VAR_TYPE.clear();
        if (!func->remove_yield())
            return false;
    }
    return true;
}

bool PTO_FUNC::remove_yield() {
    std::vector<PTO_BASE*> newStatements;

    // 记录入参的类型
    for (const auto& arg : arguments) {
        if (arg->to_string() != "self" && arg->type() != PTO_NODE_TYPE::TYPED_VARIABLE) {
            SPDLOG_ERROR("Unexpected Error for variable {} at line {}", arg->to_string(), row_);
            return false;
        }
        VAR_TYPE[arg->to_string()] = arg->get_type_str();
    }

    for (const auto& ptr : statements) {
        if (ptr->type() == PTO_NODE_TYPE::FOR_LOOP) {
            if (!gen_assign_for_for_loop_init((PTO_FOR_LOOP*)ptr, newStatements))
                return false;
        }

        if (!ptr->remove_yield())
            return false;

        newStatements.emplace_back(ptr);
    }


    statements = newStatements;

    return true;
}

void PTO_FOR_LOOP::remove_init_var() {
    for (auto& v : initVar) {
        delete v;
    }
    initVar.clear();
}

void PTO_TUPLE_VAR::clear() {
    for (auto& v : varList) {
        delete v;
    }
    varList.clear();
}

bool PTO_IF::remove_yield() {
    std::vector<PTO_BASE*> newStatements;

    for (auto& s : ifStatement) {
        if (s->type() == PTO_NODE_TYPE::FOR_LOOP) {
            if (!gen_assign_for_for_loop_init((PTO_FOR_LOOP*)s, newStatements))
                return false;
        }

        if (!s->remove_yield()) {
            return false;
        }
        
        if (s->type() == PTO_NODE_TYPE::ASSIGNMENT && ((PTO_ASSIGNMENT*)s)->get_value()->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)((PTO_ASSIGNMENT*)s)->get_value())->get_func_name() == "pypto.language.yield_") {
            if (!convert_yield_to_assign((PTO_ASSIGNMENT*)s, newStatements)) {
                return false;
            }
            delete s;
        }
        else {
            newStatements.emplace_back(s);
        }
    }

    ifStatement = newStatements;

    newStatements.clear();
    for (auto& s : elseStatement) {
        if (s->type() == PTO_NODE_TYPE::FOR_LOOP) {
            if (!gen_assign_for_for_loop_init((PTO_FOR_LOOP*)s, newStatements))
                return false;
        }

        if (!s->remove_yield()) {
            return false;
        }
        
        if (s->type() == PTO_NODE_TYPE::ASSIGNMENT && ((PTO_ASSIGNMENT*)s)->get_value()->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)((PTO_ASSIGNMENT*)s)->get_value())->get_func_name() == "pypto.language.yield_") {
            if (!convert_yield_to_assign((PTO_ASSIGNMENT*)s, newStatements)) {
                return false;
            }
            delete s;
        }
        else {
            newStatements.emplace_back(s);
        }
    }
    elseStatement = newStatements;
    
    return true;
}

bool PTO_FOR_LOOP::remove_yield() {
    std::vector<PTO_BASE*> newStatements;

    VAR_TYPE[iter->to_string()] = {"pypto.language.Scalar[pypto.language.INDEX]"};

    for (auto& s : statements) {
        if (s->type() == PTO_NODE_TYPE::FOR_LOOP) {
            if (!gen_assign_for_for_loop_init((PTO_FOR_LOOP*)s, newStatements))
                return false;
        }

        if (!s->remove_yield()) return false;

        // 是yield statememt?
        if (s->type() == PTO_NODE_TYPE::ASSIGNMENT && ((PTO_ASSIGNMENT*)s)->get_value()->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)((PTO_ASSIGNMENT*)s)->get_value())->get_func_name() == "pypto.language.yield_") {
            if (!convert_yield_to_assign((PTO_ASSIGNMENT*)s, newStatements)) {
                return false;
            }
            delete s;
        }
        else {
            newStatements.emplace_back(s);
        }
    }

    statements = newStatements;

    forLoopInitVarName.pop_back();

    return true;
}

bool PTO_ASSIGNMENT::remove_yield() {
    // 记录variable类型
    if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        VAR_TYPE[lhs->to_string()] = ((PTO_VARIABLE*)lhs)->get_type_str();
    }
    return true;
}

}