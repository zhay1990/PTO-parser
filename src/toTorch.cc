#include "ptoNode.hh"
#include "logger.hh"

namespace pto_parser{

void PTO_MODULE::dump_to_pyTorch(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');

    fout << indent << "import torch" << std::endl << std::endl;

    for (const auto& c : classOrFunc) {
        c->dump_to_pyTorch(depth, fout);
    }
}

void PTO_VARIABLE::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << varName;
    
    // 不写变量类型
}

void PTO_TUPLE_VAR::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "(";
    for (std::size_t i = 0; i < varList.size(); ++i) {
        varList[i]->dump_to_pyTorch(0, fout);
        fout << ", ";
    }
    fout << ")";
}

void PTO_LIST_VAR::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "[";
    if (varList.size() != 0) {
        varList[0]->dump_to_pyTorch(0, fout);
    }
    for (std::size_t i = 1; i < varList.size(); ++i) {
        fout << ", ";
        varList[i]->dump_to_pyTorch(0, fout);
    }
    fout << "]";
}

void PTO_FLOAT::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    fout << std::string(depth, '\t') << value;
}

void PTO_INT::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    fout << std::string(depth, '\t') << value;
}

void PTO_BOOL::dump_to_pyTorch(int, std::ofstream&) const {
    SPDLOG_ERROR("Unimplemented");
}

void PTO_BINARY_OP::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "(";
    lhs->dump_to_pyTorch(0, fout);

    switch (op) {
        case PTO_OPERATOR::ADD: fout << " + "; break;
        case PTO_OPERATOR::SUB: fout << " - "; break;
        case PTO_OPERATOR::FLOOR_DIV: fout << " // "; break;
        case PTO_OPERATOR::MUL: fout << " * "; break;
        case PTO_OPERATOR::EQUAL: fout << " == "; break;
    }
    rhs->dump_to_pyTorch(0, fout);
    fout << ")";
}

void PTO_INDEXED_VAR::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    fout << std::string(depth, '\t') << varName << "[" << index[0] << "]";
}

void PTO_CALL::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent;
    if (funcName.substr(0, 5) == "self.") {
        fout << funcName;
    }
    else if (funcName == "pypto.language.tensor.create") {
        fout << "torch.empty";
    }
    else if (funcName == "pypto.language.tensor.muls") {
        fout << "torch.mul";
    }
    else if (funcName == "pypto.language.tensor.slice") {
        // 特殊处理
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << "[";
        
        // 第二和第三个参数必须是tuple或list
        if (arguments[1]->type() != PTO_NODE_TYPE::TUPLE_VARIABLE && arguments[1]->type() != PTO_NODE_TYPE::LIST_VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[2]->type() != PTO_NODE_TYPE::TUPLE_VARIABLE && arguments[2]->type() != PTO_NODE_TYPE::LIST_VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        std::vector<PTO_BASE*> size, offset;
        if (arguments[1]->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            size = ((PTO_TUPLE_VAR*)arguments[1])->get_var_list();
        } else {
            size = ((PTO_LIST_VAR*)arguments[1])->get_var_list();
        }

        if (arguments[2]->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            offset = ((PTO_TUPLE_VAR*)arguments[2])->get_var_list();
        } else {
            offset = ((PTO_LIST_VAR*)arguments[2])->get_var_list();
        }

        if (size.size() != offset.size()) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        for (std::size_t i = 0; i < offset.size(); ++i) {
            offset[i]->dump_to_pyTorch(0, fout);
            fout << " : ";
            offset[i]->dump_to_pyTorch(0, fout);
            fout << " + ";
            size[i]->dump_to_pyTorch(0, fout);

            if (i != offset.size() - 1) {
                fout << ", ";
            }
        }
        fout << "]";


        return;
    }
    else if (funcName == "pypto.language.tensor.cast") {
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << ".to(";
        // 找到数据类型
        for (const auto& arg : arguments) {
            if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "target_type") {
                std::string targetType = ((PTO_KEYWORD*)arg)->get_value()->to_string();

                if (targetType == "pypto.language.FP32") {
                    fout << "torch.float32";
                }
                else if (targetType == "pypto.language.BF16") {
                    fout << "torch.bfloat16";
                }
                else {
                    SPDLOG_ERROR("Unimplemented target type {}", targetType);
                }
            }
        }
        fout << ")";

        return;

    }
    else if (funcName == "pypto.language.tensor.mul") {
        fout << "torch.mul";
    }
    else if (funcName == "pypto.language.tensor.row_sum") {
        fout << "torch.sum(";
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << ", dim = 1, keepdim = True)";
        return;
    }
    else if (funcName == "pypto.language.tensor.add") {
        fout << "torch.add";
    }
    else if (funcName == "pypto.language.tensor.sub") {
        fout << "torch.sub";
    }
    else if (funcName == "pypto.language.tensor.adds") {
        fout << "torch.add";
    }
    else if (funcName == "pypto.language.tensor.rsqrt") {
        fout << "torch.rsqrt";
    }
    else if (funcName == "pypto.language.tensor.row_expand_mul") {
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << " * ";
        arguments[1]->dump_to_pyTorch(0, fout);
        return;
    }
    else if (funcName == "pypto.language.tensor.col_expand_mul") {
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << " * ";
        arguments[1]->dump_to_pyTorch(0, fout);
        return;
    }
    else if (funcName == "pypto.language.tensor.matmul") {
        fout << "torch.matmul(";
        arguments[0]->dump_to_pyTorch(0, fout);
        // 是否有atrans=true的参数
        for (const auto& arg : arguments) {
            if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "a_trans" && ((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
                fout << ".mT";
            }
        }
        fout << ", ";
        arguments[1]->dump_to_pyTorch(0, fout);
        // 是否有atrans=true的参数
        for (const auto& arg : arguments) {
            if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "b_trans" && ((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
                fout << ".mT";
            }
        }
        fout << ")";
        return;
    }
    else if (funcName == "pypto.language.tensor.recip") {
        fout << "torch.reciprocal";
    }
    else if (funcName == "pypto.language.tensor.maximum") {
        fout << "torch.maximum";
    }
    else if (funcName == "pypto.language.tensor.row_max") {
        fout << "torch.max(";
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << ", dim = 1, keepdim = True)[0]";
        return;
    }
    else if (funcName == "pypto.language.tensor.row_expand_sub") {
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << " - ";
        arguments[1]->dump_to_pyTorch(0, fout);
        return;
    }
    else if (funcName == "pypto.language.tensor.row_expand_div") {
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << " / ";
        arguments[1]->dump_to_pyTorch(0, fout);
        return;
    }
    else if (funcName == "pypto.language.tensor.exp") {
        fout << "torch.exp";
    }
    else if (funcName == "pypto.language.tensor.neg") {
        fout << "torch.neg";
    }
    else if (funcName == "pypto.language.tensor.read") {
        arguments[0]->dump_to_pyTorch(0, fout);
        if (arguments[1]->type() != PTO_NODE_TYPE::LIST_VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        arguments[1]->dump_to_pyTorch(0, fout);
        return;
    }
    else if (funcName == "pypto.language.cast") {
        // 这个的转换还有些问题
        if (arguments[1]->to_string() == "pypto.language.INDEX") {
            fout << "int(";
            arguments[0]->dump_to_pyTorch(0, fout);
            fout << ")";
        }
        else {
            SPDLOG_ERROR("The target {} of cast statement at line {} is not implemented", arguments[1]->to_string(), row_);
        }
        return;
    }
    else if (funcName == "pypto.language.min") {
        fout << "min";
    }
    else if (funcName == "pypto.language.range") {
        fout << "range(";

        // 特殊处理 不输出init_values
        bool notFirstOne = false;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            if (arguments[i]->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arguments[i])->get_keyword() == "init_values") {
                continue;
            }
            if (notFirstOne) fout << ", ";
            arguments[i]->dump_to_pyTorch(0, fout);
            notFirstOne = true;
        }
        fout << ")";
        return;
    } else if (funcName == "pypto.language.parallel") {
        // 处理成串行的range
        fout << "range(";

        // 特殊处理 不输出init_values
        bool notFirstOne = false;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            if (arguments[i]->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arguments[i])->get_keyword() == "init_values") {
                continue;
            }
            if (arguments[i]->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arguments[i])->get_keyword() == "chunk") {
                continue;
            }
            if (notFirstOne) fout << ", ";
            arguments[i]->dump_to_pyTorch(0, fout);
            notFirstOne = true;
        }
        fout << ")";
        return;
    }
    else {
        SPDLOG_ERROR("Unimplemented func {}", funcName);
    }

    fout << "(";
    if (arguments.size() != 0) {
        arguments[0]->dump_to_pyTorch(0, fout);
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        fout << ", ";
        arguments[i]->dump_to_pyTorch(0, fout);
    }

    fout << ")";
}

void PTO_KEYWORD::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent (depth, '\t');
    if (keyword == "dtype") {
        fout << indent << "dtype = ";

        if (value->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (value->to_string() == "pypto.language.BF16") {
            fout << "torch.bfloat16";
        }
        else if (value->to_string() == "pypto.language.FP32") {
            fout << "torch.float32";
        }
        else {
            SPDLOG_ERROR("Unexpected dtype {}", value->to_string());
        }

    }
    else if (keyword == "layout") {
        fout << indent << "layout = torch.strided";
    }
    else {
        SPDLOG_ERROR("Unimplemented keyword {}", keyword);
    }
}

void PTO_ASSIGNMENT::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent;
    lhs->dump_to_pyTorch(0, fout); // 不写类型

    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.tensor.assemble") {
        // 这个函数特殊处理，如果第一个argument和lhs不是同一个变量，则需要拆成两个assignment
        const auto& args = ((PTO_CALL*)value)->get_arguments();

        if (args[0]->to_string() != lhs->to_string()) {
            fout << " = ";
            args[0]->dump_to_pyTorch(0, fout);
            fout << std::endl << indent;
            lhs->dump_to_pyTorch(0, fout);
        }

        fout << " [";

        std::vector<PTO_BASE*> varList;
        if (args[2]->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            varList = ((PTO_LIST_VAR*)args[2])->get_var_list();
        } else if (args[2]->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            varList = ((PTO_TUPLE_VAR*)args[2])->get_var_list();
        } else {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        for (std::size_t i = 0; i < varList.size(); ++i) {
            varList[i]->dump_to_pyTorch(0, fout);
            fout << ": ";
            varList[i]->dump_to_pyTorch(0, fout);
            fout << " + ";
            args[1]->dump_to_pyTorch(0, fout);
            fout << ".shape[" << i << "]";

            if (i != varList.size() - 1)
                fout << ", ";
        }
        
        fout << "] = ";
        args[1]->dump_to_pyTorch(0, fout);
    }
    else {
        fout << " = ";
        value->dump_to_pyTorch(0, fout);
    }

    fout << std::endl;
}

void PTO_RETURN::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "return ";
    if (returnVal.size() != 0) {
        returnVal[0]->dump_to_pyTorch(0, fout);
    }
    for (std::size_t i = 1; i < returnVal.size(); ++i) {
        fout << ", ";
        returnVal[i]->dump_to_pyTorch(0, fout);
    }
    fout << std::endl;
}

void PTO_FOR_LOOP::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "for ";
    iter->dump_to_pyTorch(0, fout);
    fout << " in ";
    info->dump_to_pyTorch(0, fout);
    fout << ":" << std::endl;

    for (const auto& s : statements) {
        s->dump_to_pyTorch(depth + 1, fout);
    }
}

void PTO_IF::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "if ";
    comparator->dump_to_pyTorch(0, fout);
    fout << ":" << std::endl;
    if (ifStatement.size() == 0) {
        fout << indent << "\tpass";
    }
    for (const auto& s : ifStatement) {
        s->dump_to_pyTorch(depth + 1, fout);
    }
    if (elseStatement.size() != 0) {
        fout << indent << "else:" << std::endl;
    }
    for (const auto& s : elseStatement) {
        s->dump_to_pyTorch(depth + 1, fout);
    } 
}

void PTO_FUNC::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "def " << funcName << "(";
    if (arguments.size() != 0) {
        arguments[0]->dump_to_pyTorch(0, fout);
    }
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        fout << ", ";
        arguments[i]->dump_to_pyTorch(0, fout);
    }
    fout << "):" << std::endl;
    // 不输出返回类型

    for (const auto& s : statements) {
        s->dump_to_pyTorch(depth + 1, fout);
    }
}

void PTO_CLASS::dump_to_pyTorch(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << "class " << name << "_torch:" << std::endl;

    for (const auto& f : functions) {
        f->dump_to_pyTorch(depth + 1, fout);
    }
}


} // namespace pto_parser
