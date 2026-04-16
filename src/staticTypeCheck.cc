#include "ptoNode.hh"
#include "logger.hh"
namespace pto_parser{

typedef std::unordered_map<std::string, PTO_TYPE> STR_PTO_TYPE_MAP;

// 辅助函数
static bool add_var_to_map(PTO_BASE* var, STR_PTO_TYPE_MAP& validVar) {
    // 根据var的类型处理
    if (var->type() == PTO_NODE_TYPE::VARIABLE || var->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (var->get_data_type().kind == PTO_TYPE_KIND::UNKNOWN) {
            SPDLOG_ERROR("Unexpected call of add_var_to_map for {} at line {}", var->to_string(), var->row());
            return false;
        }

        if (validVar.find(var->to_string()) != validVar.end()) {
            if (validVar[var->to_string()].isDynamic) {
                var->get_data_type().isDynamic = true;
                validVar[var->to_string()] = var->get_data_type();
            }
            else {
                // 应当一致
                if (var->get_data_type() != validVar[var->to_string()]) {
                    SPDLOG_ERROR("Mismatched data type between {} and {} for variable {}",
                        var->get_data_type().to_string(),
                        validVar[var->to_string()].to_string(),
                        var->to_string() 
                    );
                    return false;
                }
            }
        } else {
            // 添加
            validVar[var->to_string()] = var->get_data_type();
        }
    }
    else if (var->type() == PTO_NODE_TYPE::LIST_VARIABLE || var->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
        SPDLOG_INFO(var->to_string());
        SPDLOG_INFO(var->get_data_type().to_string());
        if (var->get_data_type().kind != PTO_TYPE_KIND::TUPLE) {
            SPDLOG_ERROR("Unexpected call of add_var_to_map");
            return false;
        }

        // 每一个var单独添加
        std::vector<PTO_BASE*> varList;
        if (var->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            varList = ((PTO_LIST_VAR*)var)->get_var_list();
        } else {
            varList = ((PTO_TUPLE_VAR*)var)->get_var_list();
        }

        if (varList.size() != var->get_data_type().sub_types.size()) {
            SPDLOG_ERROR("Unexpected Error");
            return false;
        }

        for (std::size_t i = 0; i < varList.size(); ++i) {
            const auto& var = varList[i];
            if (var->get_data_type().kind == PTO_TYPE_KIND::UNKNOWN) {
                SPDLOG_ERROR("Unexpected call of add_var_to_map");
                return false;
            }

            // 类型应当都是variable
            if (var->type() != PTO_NODE_TYPE::VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }

            if (validVar.find(var->to_string()) != validVar.end()) {
                // 已经有该变量的定义，先继承isDynamic的定义
                var->get_data_type().isDynamic = validVar[var->to_string()].isDynamic;

                // 应当一致
                if (var->get_data_type() != validVar[var->to_string()]) {
                    SPDLOG_ERROR("Mismatched data type between {} and {} for variable {}",
                        var->get_data_type().to_string(),
                        validVar[var->to_string()].to_string(),
                        var->to_string() 
                    );
                    return false;
                }
            } else {
                // 添加
                validVar[var->to_string()] = var->get_data_type();
            }
        }
    }
    else {
        SPDLOG_ERROR("Unexpected Call of add_var_to_map");
        return false;
    }

    return true;
}

static PTO_TYPE parse_type_str(const std::string& str, const STR_PTO_TYPE_MAP& validVar, const uint32_t& row_) {
    PTO_TYPE ret;
    if (str.find("pypto.language.Tensor") == 0) {
        // Tensor
        std::string content = str.substr(str.find_first_of('[') + 1);
        content.pop_back();

        // 第一个应当是dimension
        std::string dimension = content.substr(1, content.find_first_of(']') - 1);
        content = content.substr(content.find_first_of(']') + 2);

        std::vector<int> shape;
        std::vector<std::string> dynamicShape;
        while (true) {
            size_t index = dimension.find_first_of(',');
            if (index == std::string::npos) {
                // 除去开头和结尾的空格
                dimension = dimension.substr(dimension.find_first_not_of(' '));
                while (dimension.size() > 0 && dimension.back() == ' ') dimension.pop_back();

                try {
                    shape.emplace_back(std::stoi(dimension));
                }
                catch (const std::invalid_argument& e) {
                    shape.emplace_back(-1);
                    
                    // 这个变量应当存在于validVar中，且属性是dynamic
                    if (validVar.find(dimension) == validVar.end()) {
                        SPDLOG_ERROR("Undefined variable {} at line {}", dimension, row_);
                    }
                    else if (!validVar.find(dimension)->second.isDynamic) {
                        SPDLOG_ERROR("Variable {} at line is not defined as a dynamic", dimension, row_);
                    }
                    else {
                        dynamicShape.emplace_back(dimension);
                    }
                }
                break;
            } else {
                std::string temp = dimension.substr(0, index);
                temp = temp.substr(temp.find_first_not_of(' '));
                while (temp.size() > 0 && temp.back() == ' ') temp.pop_back();

                try {
                    shape.emplace_back(std::stoi(temp));
                } 
                catch (const std::invalid_argument& e) {
                    shape.emplace_back(-1);
                    
                    // 这个变量应当存在于validVar中，且属性是dynamic
                    if (validVar.find(temp) == validVar.end()) {
                        SPDLOG_ERROR("Undefined variable {} at line {}", temp, row_);
                    }
                    else if (!validVar.find(temp)->second.isDynamic) {
                        SPDLOG_ERROR("Variable {} at line is not defined as a dynamic", temp, row_);
                    }
                    else {
                        dynamicShape.emplace_back(temp);
                    }
                }
                dimension = dimension.substr(index + 1);
            }
        }
        // 第二个是数据类型
        std::string tensorType = content;
        if (content.find_first_of(',') == std::string::npos) {
            content = "";
        } else {
            tensorType = content.substr(0, content.find_first_of(','));
            content = content.substr(content.find_first_of(',') + 1);
        }

        if (tensorType == "pypto.language.BF16") {
            ret = PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::BF16);
        }
        else if (tensorType == "pypto.language.FP32") {
            ret = PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::FP32);
        }
        else if (tensorType == "pypto.language.INT32") {
            ret = PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::INT32);
        }
        else {
            SPDLOG_ERROR("Unimplemented data type '{}'", tensorType);
        }

        ret.dynamicShape = dynamicShape;

        if (content != "") {
            SPDLOG_WARN("'{}' in type string is ignored", content);
        }
    }
    else if (str == "pypto.language.Scalar[pypto.language.INDEX]") {
        ret = PTO_TYPE::make_scalar(PTO_TYPE_KIND::INT32);
    }
    else if (str == "pypto.language.Scalar[pypto.language.INT32]") {
        ret = PTO_TYPE::make_scalar(PTO_TYPE_KIND::INT32);
    }
    else {
        SPDLOG_ERROR("Unprocessed data type {}", str);
    }
    return ret;
}

void PTO_VARIABLE::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    // 类型推导
    if (typeStr.size() == 0) {
        if (validVar.find(varName) != validVar.end()) {
            dataType = validVar.find(varName)->second;
        }
        else {
            SPDLOG_ERROR("Undefined variable {} at line {}", varName, row_);
        }
    }
    else if (typeStr.size() == 1) {
        this->dataType = parse_type_str(typeStr[0], validVar, row_);
    }
    else if (typeStr.size() > 1) {
        // 处理为tuple
        std::vector<PTO_TYPE> types;
        for (const auto& str : typeStr) {
            types.emplace_back(parse_type_str(str, validVar, row_));
        }
        this->dataType = PTO_TYPE::make_tuple(types);
    }
}

void PTO_TUPLE_VAR::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    // 推导包含的参数的类型
    std::vector<PTO_TYPE> tupleType;
    for (const auto& ptr : varList) {
        ptr->infer_type(validVar);
        tupleType.emplace_back(ptr->get_data_type().kind);
    }

    dataType = PTO_TYPE::make_tuple(tupleType);
}

void PTO_LIST_VAR::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    // 推导包含的参数的类型
    std::vector<PTO_TYPE> tupleType;
    for (const auto& ptr : varList) {
        ptr->infer_type(validVar);
        tupleType.emplace_back(ptr->get_data_type().kind);
    }

    dataType = PTO_TYPE::make_tuple(tupleType);
}

void PTO_FLOAT::infer_type(STR_PTO_TYPE_MAP&) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_FLOAT);
}

void PTO_INT::infer_type(STR_PTO_TYPE_MAP&) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_INT);
}

void PTO_BOOL::infer_type(STR_PTO_TYPE_MAP&) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_BOOL);
}


void PTO_BINARY_OP::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    // 先推断lhs和rhs的类型
    lhs->infer_type(validVar);
    rhs->infer_type(validVar);

    // 获取推断出的具体类型
    const PTO_TYPE& l_type = lhs->get_data_type();
    const PTO_TYPE& r_type = rhs->get_data_type();

    // 如果任一子节点类型未知，则当前节点也是未知，无法继续推断
    if (l_type.kind == PTO_TYPE_KIND::UNKNOWN || r_type.kind == PTO_TYPE_KIND::UNKNOWN) {
        dataType = PTO_TYPE(PTO_TYPE_KIND::UNKNOWN);
        return;
    }

    // ==========================================
    // 核心特征提取：正交化区分"运行时/常量" 与 "浮点/整数"
    // ==========================================
    bool is_l_runtime = (l_type.kind == PTO_TYPE_KIND::FP32 || l_type.kind == PTO_TYPE_KIND::BF16 || l_type.kind == PTO_TYPE_KIND::INT32);
    bool is_r_runtime = (r_type.kind == PTO_TYPE_KIND::FP32 || r_type.kind == PTO_TYPE_KIND::BF16 || r_type.kind == PTO_TYPE_KIND::INT32);
    bool is_runtime   = is_l_runtime || is_r_runtime; // 只要有一个是运行时，结果就是运行时

    bool is_l_float   = (l_type.kind == PTO_TYPE_KIND::FP32 || l_type.kind == PTO_TYPE_KIND::BF16 || l_type.kind == PTO_TYPE_KIND::CONST_FLOAT);
    bool is_r_float   = (r_type.kind == PTO_TYPE_KIND::FP32 || r_type.kind == PTO_TYPE_KIND::BF16 || r_type.kind == PTO_TYPE_KIND::CONST_FLOAT);
    bool is_float     = is_l_float || is_r_float;     // 只要有一个是浮点，结果就是浮点

    // 根据操作符类型推断 dataType
    switch (op) {
        case PTO_OPERATOR::ADD:
        case PTO_OPERATOR::SUB:
        case PTO_OPERATOR::MUL: {
            // 1. 拦截 Tensor 类型并报错 (加/减/乘 均不支持)
            if (l_type.kind == PTO_TYPE_KIND::TENSOR || r_type.kind == PTO_TYPE_KIND::TENSOR) {
                std::string op_str = (op == PTO_OPERATOR::ADD) ? "+" : (op == PTO_OPERATOR::SUB ? "-" : "*");
                SPDLOG_ERROR("Type Error at line {}: Operator '{}' does not support TENSOR types.", row_, op_str);
                dataType = PTO_TYPE(PTO_TYPE_KIND::UNKNOWN);
                break; // 拦截后直接跳出，不再向下推断
            }

            // 2. 标量类型推断矩阵
            if (is_float) {
                if (is_runtime) {
                    // 运行时浮点类型提升细节
                    if (l_type.kind == PTO_TYPE_KIND::FP32 || r_type.kind == PTO_TYPE_KIND::FP32) {
                        dataType = PTO_TYPE(PTO_TYPE_KIND::FP32);
                    } else if (l_type.kind == PTO_TYPE_KIND::BF16 || r_type.kind == PTO_TYPE_KIND::BF16) {
                        dataType = PTO_TYPE(PTO_TYPE_KIND::BF16);
                    } else {
                        // 特殊情况：CONST_FLOAT + INT32 等，默认提升为运行时的 FP32
                        dataType = PTO_TYPE(PTO_TYPE_KIND::FP32);
                    }
                } else {
                    // 纯常量浮点计算：CONST_FLOAT + CONST_INT 等
                    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_FLOAT);
                }
            } else {
                if (is_runtime) {
                    // 运行时整数计算：CONST_INT + INT32 等
                    dataType = PTO_TYPE(PTO_TYPE_KIND::INT32);
                } else {
                    // 纯常量整数计算：CONST_INT + CONST_INT
                    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_INT);
                }
            }
            break;
        }

        case PTO_OPERATOR::FLOOR_DIV: {
            // 1. 拦截 Tensor 类型并报错
            if (l_type.kind == PTO_TYPE_KIND::TENSOR || r_type.kind == PTO_TYPE_KIND::TENSOR) {
                SPDLOG_ERROR("Type Error at line {}: Operator '//' does not support TENSOR types.", row_);
                dataType = PTO_TYPE(PTO_TYPE_KIND::UNKNOWN);
            } else {
                // 2. 地板除强制返回整型，只需区分是否为常量
                if (is_runtime) {
                    dataType = PTO_TYPE(PTO_TYPE_KIND::INT32);
                } else {
                    dataType = PTO_TYPE(PTO_TYPE_KIND::CONST_INT);
                }
            }
            break;
        }

        case PTO_OPERATOR::EQUAL: {
            // 1. 拦截 Tensor 类型并报错
            if (l_type.kind == PTO_TYPE_KIND::TENSOR || r_type.kind == PTO_TYPE_KIND::TENSOR) {
                SPDLOG_ERROR("Type Error at line {}: Operator '==' does not support TENSOR types.", row_);
                dataType = PTO_TYPE(PTO_TYPE_KIND::UNKNOWN);
            } else {
                // 2. 目前只返回编译期常量 BOOL
                dataType = PTO_TYPE(PTO_TYPE_KIND::BOOL);
            }
            break;
        }
    }
}

void PTO_INDEXED_VAR::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    if (validVar.find(varName) == validVar.end()) {
        SPDLOG_ERROR("Undefinited variable {} at line {}", varName, row_);
        return;
    }

    if (validVar[varName].kind != PTO_TYPE_KIND::TUPLE) {
        SPDLOG_ERROR("The indexed variable should have tuple data type, but get {}", validVar[varName].to_string());
        return;
    }

    if (index.size() != 1) {
        SPDLOG_ERROR("Indexed variable {} at line {} is one dimension.", varName, row_);
        return;
    }

    if ((int)validVar[varName].sub_types.size() <= index[0]) {
        SPDLOG_ERROR("Index ({}) out of range ({}) for variable {} at line {}", 
            index[0],
            validVar[varName].sub_types.size(),
            varName, row_
        );
        return;
    }

    this->dataType = validVar[varName].sub_types[index[0]];
}

void PTO_CALL::infer_type(STR_PTO_TYPE_MAP& validVar) {
    if (dataType.kind != PTO_TYPE_KIND::UNKNOWN) return;

    // 对Intrinsic函数的处理
    if (funcName == "pypto.language.tensor.create") {
        // 应当有三组arguments
        if (arguments.size() != 3) {
            SPDLOG_ERROR("Unexpected Error");
        }

        // 第一个参数应当是个tuple
        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TUPLE) {
            SPDLOG_ERROR("Unexpected Error");
        }

        // 第一个参数内部都应当是整数
        for (const auto& t : arguments[0]->get_data_type().sub_types) {
            if (t != PTO_TYPE_KIND::CONST_INT) {
                SPDLOG_ERROR("Unexpected Error");
            }
        }

        // 拿到shape
        std::vector<int> shape;
        for (const auto& e : ((PTO_LIST_VAR*)arguments[0])->get_var_list()) {
            shape.emplace_back(((PTO_INT*)e)->get_value());
        }

        // 第二个参量必须是dType
        if (arguments[1]->type() != PTO_NODE_TYPE::KEYWORD) {
            SPDLOG_ERROR("Unexpected Error");
        }
        PTO_KEYWORD* dType = (PTO_KEYWORD*)arguments[1];
        if (dType->get_keyword() != "dtype") {
            SPDLOG_ERROR("Unexpected Error");
        }
        if (dType->get_value()->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
        }
        if (dType->get_value()->to_string() == "pypto.language.BF16") {
            dataType = PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::BF16);
        }
        else if (dType->get_value()->to_string() == "pypto.language.FP32") {
            dataType = PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::FP32);
        }
        else {
            SPDLOG_ERROR("Unsupported datatype '{}'", dType->get_value()->to_string());
        }

        // 第三个参量必须是layout
        if (arguments[2]->type() != PTO_NODE_TYPE::KEYWORD) {
            SPDLOG_ERROR("Unexpected Error");
        }
        PTO_KEYWORD* layout = (PTO_KEYWORD*)arguments[2];
        if (layout->get_keyword() != "layout") {
            SPDLOG_ERROR("Unexpected Error");
        }
        if (layout->get_value()->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
        }
        if (layout->get_value()->to_string() != "pypto.language.TensorLayout.ND") {
            SPDLOG_WARN("Unsupported layout '{}', change it to pypto.language.TensorLayout.ND", layout->get_value()->to_string());
        }

    }
    else if (funcName == "pypto.language.tensor.muls") {
        // 只有两个参数
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
        }

        // 解析类型
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 分几种情况分析
        if (arguments[0]->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
            if (arguments[1]->get_data_type().kind == PTO_TYPE_KIND::CONST_FLOAT || arguments[1]->get_data_type().kind == PTO_TYPE_KIND::CONST_INT)
                dataType = arguments[0]->get_data_type();
        }
        else {
            SPDLOG_ERROR("Unprocessed muls between {} and {}", arguments[0]->get_data_type().to_string(), arguments[1]->get_data_type().to_string());
        }
    }
    else if (funcName == "pypto.language.tensor.slice") {
        if (arguments.size() < 3) {
            SPDLOG_ERROR("Unexpected Error");
        }

        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Wrong input for slice");
            return;
        }

        if (arguments[1]->type() != PTO_NODE_TYPE::LIST_VARIABLE) {
            SPDLOG_ERROR("Unexepcted Error");
            return;
        }

        PTO_LIST_VAR *ptr = (PTO_LIST_VAR*)arguments[1];
        const auto& varList = ptr->get_var_list();

        // 应当都是PTO_INT
        std::vector<int> shape;
        std::vector<std::string> dynamicShape;
        for (const auto& var : varList) {
            if (var->type() == PTO_NODE_TYPE::INT_CONSTANT) {
                shape.emplace_back(((PTO_INT*)var)->get_value());
            }
            else if (var->type() == PTO_NODE_TYPE::VARIABLE) {
                shape.emplace_back(-1);
                if (validVar.find(var->to_string()) == validVar.end()) {
                    SPDLOG_ERROR("Undefined variable {} at line {}", var->to_string(), var->row());
                }
                else if (!validVar[var->to_string()].isDynamic) {
                    SPDLOG_ERROR("Variable {} at line {} is not a dynamic", var->to_string(), var->row());
                }
                else {
                    dynamicShape.emplace_back(var->to_string());
                }
            }
        }

        dataType = arguments[0]->get_data_type();

        if (dataType.shape.size() != shape.size()) {
            SPDLOG_ERROR("Unmatched data type for slice at line {}", row_);
            return;
        }
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] > dataType.shape[i]) {
                SPDLOG_ERROR("Unmatched data type for slice at line {}", row_);
                return;
            }
        }
        dataType.shape = shape;
        dataType.dynamicShape = dynamicShape;
    }
    else if (funcName == "pypto.language.tensor.cast") {
        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        dataType = arguments[0]->get_data_type();

        // 找到target_type的keyword
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            if (arguments[i]->type() != PTO_NODE_TYPE::KEYWORD)
                continue;
            PTO_KEYWORD* ptr = (PTO_KEYWORD*)arguments[i];

            if (ptr->get_keyword() != "target_type")
                continue;
            
            if (ptr->get_value()->type() != PTO_NODE_TYPE::VARIABLE) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            if (ptr->get_value()->to_string() == "pypto.language.FP32") {
                dataType.sub_types[0] = PTO_TYPE_KIND::FP32;
            }
            else if (ptr->get_value()->to_string() == "pypto.language.BF16") {
                dataType.sub_types[0] = PTO_TYPE_KIND::BF16;
            }
            else {
                SPDLOG_ERROR("Unprocessed data type '{}' in cast", ptr->get_value()->to_string());
            }
        }
    } 
    else if (funcName == "pypto.language.tensor.mul") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[1]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 强制要求两个tensor类型相同
        if (arguments[0]->get_data_type() != arguments[1]->get_data_type()) {
            SPDLOG_ERROR("Mismatched type for mul: {} vs {}", arguments[0]->get_data_type().to_string(), arguments[1]->get_data_type().to_string());
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.row_sum") {
        arguments[0]->infer_type(validVar);

        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        std::vector<int> shape = arguments[0]->get_data_type().shape;
        shape[1] = 1;
        dataType = PTO_TYPE::make_tensor(shape, arguments[0]->get_data_type().sub_types[0]);
    }
    else if (funcName == "pypto.language.tensor.add" || funcName == "pypto.language.tensor.sub") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[1]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 两个tensor的shape应当相同
        if (arguments[0]->get_data_type().shape != arguments[1]->get_data_type().shape) {
            SPDLOG_ERROR("Mismatched type for add: {} vs {}", arguments[0]->get_data_type().to_string(), arguments[1]->get_data_type().to_string());
            return;
        }

        dataType = arguments[0]->get_data_type();

        // 两个tensor的数据类型可能不一样，取精度高的
        if (arguments[0]->get_data_type().sub_types[0] == PTO_TYPE_KIND::FP32 || arguments[1]->get_data_type().sub_types[0] == PTO_TYPE_KIND::FP32) {
            dataType.sub_types[0] = PTO_TYPE_KIND::FP32;
        }
    }
    else if (funcName == "pypto.language.tensor.adds") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 第一个是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.rsqrt") {
        arguments[0]->infer_type(validVar);

        // 第一个是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.row_expand_mul") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都应当是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 只支持二维tensor
        if(arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        // row应当相等
        if (arguments[0]->get_data_type().shape[0] != arguments[1]->get_data_type().shape[0]) {
            SPDLOG_ERROR("The number of row should be equal for row_expand_mul at line {}", row_);
            return;
        }

        if (arguments[0]->get_data_type().shape[1] == 1) {
            this->dataType = arguments[1]->get_data_type();
        } else {
            this->dataType = arguments[0]->get_data_type();
        }
    }
    else if (funcName == "pypto.language.tensor.col_expand_mul") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都应当是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 只支持二维tensor
        if(arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        // column应当相等
        if (arguments[0]->get_data_type().shape[1] != arguments[1]->get_data_type().shape[1]) {
            SPDLOG_ERROR("The number of column should be equal for row_expand_mul at line {}", row_);
            return;
        }

        if (arguments[0]->get_data_type().shape[0] == 1) {
            this->dataType = arguments[1]->get_data_type();
        } else {
            this->dataType = arguments[0]->get_data_type();
        }
    }
    else if (funcName == "pypto.language.tensor.row_expand_sub") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都应当是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 只支持二维tensor
        if(arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        // row应当相等
        if (arguments[0]->get_data_type().shape[0] != arguments[1]->get_data_type().shape[0]) {
            SPDLOG_ERROR("The number of row should be equal for row_expand_mul at line {}", row_);
            return;
        }

        if (arguments[0]->get_data_type().shape[1] == 1) {
            this->dataType = arguments[1]->get_data_type();
        } else {
            this->dataType = arguments[0]->get_data_type();
        }
    }
    else if (funcName == "pypto.language.tensor.row_expand_div") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都应当是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 只支持二维tensor
        if(arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        // row应当相等
        if (arguments[0]->get_data_type().shape[0] != arguments[1]->get_data_type().shape[0]) {
            SPDLOG_ERROR("The number of row should be equal for row_expand_mul at line {}", row_);
            return;
        }

        if (arguments[0]->get_data_type().shape[1] == 1) {
            this->dataType = arguments[1]->get_data_type();
        } else {
            this->dataType = arguments[0]->get_data_type();
        }
    }
    else if (funcName == "pypto.language.tensor.matmul") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都应当是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 两个的数据类型应当相同
        if (arguments[0]->get_data_type().sub_types[0] != arguments[1]->get_data_type().sub_types[0]) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 只支持二维tensor
        if(arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        std::vector<int> shape = arguments[0]->get_data_type().shape;

        // 是否有a_trans的keyword？
        for (const auto& arg : arguments) {
            if (arg->type() != PTO_NODE_TYPE::KEYWORD)
                continue;
            PTO_KEYWORD *ptr = (PTO_KEYWORD*)arg;

            if (ptr->get_keyword() != "a_trans")
                continue;
            
            if (ptr->get_value()->type() != PTO_NODE_TYPE::BOOL_CONSTANT) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            if (ptr->get_value()->to_string() == "true") {
                // 转置A矩阵
                shape[0] = arguments[0]->get_data_type().shape[1];
                shape[1] = arguments[0]->get_data_type().shape[0];
            }
        }

        // 是否有b_trans的keyword？
        for (const auto& arg : arguments) {
            if (arg->type() != PTO_NODE_TYPE::KEYWORD)
                continue;
            PTO_KEYWORD *ptr = (PTO_KEYWORD*)arg;

            if (ptr->get_keyword() != "b_trans")
                continue;
            
            if (ptr->get_value()->type() != PTO_NODE_TYPE::BOOL_CONSTANT) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            if (ptr->get_value()->to_string() == "true") {
                // 按B矩阵做了转置做检查
                if (shape[1] != arguments[1]->get_data_type().shape[1]) {
                    SPDLOG_ERROR("Mismatched tensor shape for matmul at line {}, {} vs {}",
                        row_,
                        arguments[0]->get_data_type().to_string(),
                        arguments[1]->get_data_type().to_string()
                    );
                    return;
                }

                shape[1] = arguments[1]->get_data_type().shape[0];
            } else {
                if (shape[1] != arguments[1]->get_data_type().shape[0]) {
                    SPDLOG_ERROR("Mismatched tensor shape for matmul at line {}, {} vs {}",
                        row_,
                        arguments[0]->get_data_type().to_string(),
                        arguments[1]->get_data_type().to_string()
                    );
                    return;
                }

                shape[1] = arguments[1]->get_data_type().shape[1];
            }
        }

        this->dataType = PTO_TYPE::make_tensor(shape, arguments[0]->get_data_type().sub_types[0]);
    }
    else if (funcName == "pypto.language.tensor.assemble") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);
        arguments[2]->infer_type(validVar);

        // 做些静态检查
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[1]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().sub_types[0] != arguments[1]->get_data_type().sub_types[0]) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().shape.size() != 2 || arguments[1]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimensional tensor in assemble");
            return;
        }

        if (arguments[0]->get_data_type().shape[0] < arguments[1]->get_data_type().shape[0] || arguments[0]->get_data_type().shape[1] < arguments[1]->get_data_type().shape[1]) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        this->dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.row_max") {
        // 应当只有一个argmuent
        if (arguments.size() != 1) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimensional tensor");
            return;
        }

        std::vector<int> shape = arguments[0]->get_data_type().shape;
        shape[1] = 1;

        dataType = PTO_TYPE::make_tensor(shape, arguments[0]->get_data_type().sub_types[0]);
    }
    else if (funcName == "pypto.language.tensor.exp") {
        // 应当只有一个argmuent
        if (arguments.size() != 1) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimensional tensor");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.neg") {
        // 应当只有一个argmuent
        if (arguments.size() != 1) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimensional tensor");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.recip") {
        // 应当只有一个argmuent
        if (arguments.size() != 1) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[0]->get_data_type().shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimensional tensor");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.maximum") {
        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        // 两个都是tensor
        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR || arguments[1]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 两个类型相同
        if (arguments[0]->get_data_type() != arguments[1]->get_data_type()) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        dataType = arguments[0]->get_data_type();
    }
    else if (funcName == "pypto.language.tensor.read") {
        // 应当有两个输入
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);

        if (arguments[0]->get_data_type().kind != PTO_TYPE_KIND::TENSOR) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        this->dataType = PTO_TYPE::make_scalar(arguments[0]->get_data_type().sub_types[0].kind);
    }
    else if (funcName == "pypto.language.yield_") {
        std::vector<PTO_TYPE> types;
        for (const auto& arg : arguments) {
            arg->infer_type(validVar);
            types.emplace_back(arg->get_data_type());
        }
        dataType = PTO_TYPE::make_tuple(types);
    }
    else if (funcName == "pypto.language.min") {
        // 应当只有两个arguments
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);
        arguments[1]->infer_type(validVar);

        switch (arguments[0]->get_data_type().kind) {
            case PTO_TYPE_KIND::FP32: case PTO_TYPE_KIND::BF16: case PTO_TYPE_KIND::CONST_FLOAT:
            case PTO_TYPE_KIND::INT32: case PTO_TYPE_KIND::CONST_INT:
            break;
            default:
                SPDLOG_ERROR("Unexpected data type {} for min", arguments[0]->get_data_type().to_string());
                break;
        }

        switch (arguments[1]->get_data_type().kind) {
            case PTO_TYPE_KIND::FP32: case PTO_TYPE_KIND::BF16: case PTO_TYPE_KIND::CONST_FLOAT:
            case PTO_TYPE_KIND::INT32: case PTO_TYPE_KIND::CONST_INT:
            break;
            default:
                SPDLOG_ERROR("Unexpected data type {} for min", arguments[1]->get_data_type().to_string());
                break;
        }

        if (arguments[0]->get_data_type().kind == PTO_TYPE_KIND::FP32 || arguments[1]->get_data_type().kind == PTO_TYPE_KIND::FP32) {
            this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::FP32);
        }
        else if (arguments[0]->get_data_type().kind == PTO_TYPE_KIND::BF16 || arguments[1]->get_data_type().kind == PTO_TYPE_KIND::BF16) {
            this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::BF16);
        }
        else if (arguments[0]->get_data_type().kind == PTO_TYPE_KIND::CONST_FLOAT || arguments[1]->get_data_type().kind == PTO_TYPE_KIND::CONST_FLOAT) {
            this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::FP32);
        }
        else if (arguments[0]->get_data_type().kind == PTO_TYPE_KIND::INT32 || arguments[1]->get_data_type().kind == PTO_TYPE_KIND::INT32) {
            this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::INT32);
        }
        else {
            this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::CONST_INT);
        }
    }
    else if (funcName == "pypto.language.cast") {
        // 应当只有两个arguments
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        arguments[0]->infer_type(validVar);

        // 第二arg应当是variable
        if (arguments[1]->type() != PTO_NODE_TYPE::VARIABLE) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        if (arguments[1]->to_string() == "pypto.language.INDEX") {
            switch (arguments[0]->get_data_type().kind) {
                case PTO_TYPE_KIND::CONST_INT: case PTO_TYPE_KIND::CONST_FLOAT:
                    this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::CONST_INT);
                    break;
                case PTO_TYPE_KIND::INT32: case PTO_TYPE_KIND::BF16: case PTO_TYPE_KIND::FP32:
                    this->dataType = PTO_TYPE::make_scalar(PTO_TYPE_KIND::INT32);
                    break;
                default:
                    SPDLOG_ERROR("Unexpected data type {} for cast function", arguments[0]->get_data_type().to_string());
                    break;
            }
        }
        else {
            SPDLOG_ERROR("Unimplemented process method for {}", arguments[1]->to_string());
            return;
        }
    }
    else if (funcName.substr(0, 5) == "self." && validVar.find(funcName.substr(5)) != validVar.end()) {
        // 当前只能处理self.xxx的用户定义函数调用
        this->dataType = validVar.find(funcName.substr(5))->second;
    }
    else {
        SPDLOG_ERROR("Process method for function call '{}' is not implemented", funcName);
    }
}

bool PTO_ASSIGNMENT::type_check(STR_PTO_TYPE_MAP& validVar) {
    // 先解析rhs
    switch (value->type()) {
        case PTO_NODE_TYPE::FUNC_CALL: // 函数调用
        // dynamic函数做特殊处理
        if (((PTO_CALL*)value)->get_func_name() == "pypto.language.dynamic") {
            // 此时左值的类型还未推导出来，但需要将其设置为dynamic
            if (lhs->get_data_type().kind != PTO_TYPE_KIND::UNKNOWN) {
                SPDLOG_ERROR("Unexpected Error");
            }

            lhs->get_data_type().isDynamic = true;

            // 强制要求validMap中没有该定义
            if (validVar.find(lhs->to_string()) != validVar.end()) {
                SPDLOG_ERROR("Unexpected Error");
            }

            validVar[lhs->to_string()] = lhs->get_data_type();
        }
        // 剩下的根据左值类型做推断
        else if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            // 推断左值类型
            lhs->infer_type(validVar);

            // 将左值添加到Map中
            if (!add_var_to_map(lhs, validVar)) return false;

            // 推断函数返回类型
            value->infer_type(validVar);

            if (value->get_data_type().kind == PTO_TYPE_KIND::TUPLE && value->get_data_type().sub_types.size() == 1) {
                // 实际只有一个type
                value->get_data_type() = value->get_data_type().sub_types[0];
            }

            if (lhs->get_data_type() != value->get_data_type()) {
                // 有可能仅仅是数据类型不一致
                if (lhs->get_data_type().kind == value->get_data_type().kind && lhs->get_data_type().shape == value->get_data_type().shape && lhs->get_data_type().dynamicShape == value->get_data_type().dynamicShape) {
                    if (lhs->get_data_type().sub_types[0] == PTO_TYPE_KIND::FP32) {
                        SPDLOG_WARN("Implicit data type convert from {} to {} at line {}",
                            value->get_data_type().to_string(),
                            lhs->get_data_type().to_string(),
                            row_
                        );
                        return true;
                    }   
                }

                SPDLOG_ERROR("Mismatched data type at line {}, {} vs {}", row_, lhs->get_data_type().to_string(), value->get_data_type().to_string());
                return false;
            }
        }
        else if (lhs->type() == PTO_NODE_TYPE::VARIABLE) {
            // 推断函数返回类型
            value->infer_type(validVar);

            if (value->get_data_type().kind == PTO_TYPE_KIND::TUPLE && value->get_data_type().sub_types.size() == 1) {
                // 实际只有一个type
                value->get_data_type() = value->get_data_type().sub_types[0];
            }
            
            lhs->get_data_type() = value->get_data_type();

            if (!add_var_to_map(lhs, validVar))
                return false;
        }
        else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE || lhs->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            std::vector<PTO_BASE*> varList;
            if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
                varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            } else {
                varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            }

            // 每个variable都应当是variable类型
            for (const auto& var : varList) {
                if (var->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_ERROR("Unexpected Error");
                    return false;
                }
            }

            // 推断函数返回值
            value->infer_type(validVar);

            // 根据函数返回值为左值赋予类型
            if (value->get_data_type().kind != PTO_TYPE_KIND::TUPLE) {
                SPDLOG_ERROR("Unexpected Error");
                return false;
            }
            const auto& types = value->get_data_type().sub_types;

            if (varList.size() != types.size()) {
                SPDLOG_ERROR("Mismatched number of element for assignment at line {}, {} vs {}",
                    row_,
                    varList.size(),
                    types.size()
                );
                return false;
            }
            for (std::size_t i = 0; i < varList.size(); ++i) {
                varList[i]->get_data_type() = types[i];

                if (!add_var_to_map(varList[i], validVar))
                    return false;
            }
        }
        else {
            SPDLOG_ERROR("Unexpected lhs type for assignment at line {}", row_);
            return false;
        }

        break;

        case PTO_NODE_TYPE::BINARY_OP:
        case PTO_NODE_TYPE::VARIABLE:
        case PTO_NODE_TYPE::INDEXED_VARIABLE:
        
        // 推断左值和右值的类型
        lhs->infer_type(validVar);
        value->infer_type(validVar);

        // 两者类型应当一致
        if (lhs->get_data_type() != value->get_data_type()) {
            SPDLOG_ERROR("Mismatched data type for assignment at line {}: {} vs {}",
                row_,
                lhs->get_data_type().to_string(),
                value->get_data_type().to_string()
            );
            return false;
        }

        if (!add_var_to_map(lhs, validVar)) return false;

        break;

        default:
        SPDLOG_ERROR("Unimplemented assignment at line {}", row_);
        return false;
    }
    
    return true;
}

void PTO_RETURN::infer_type(STR_PTO_TYPE_MAP& validVar) {
    // 当前假定return的都是变量名，后续是情况拓展
    std::vector<PTO_TYPE> ret;
    for (const auto& r : returnVal) {
        r->infer_type(validVar);
        ret.emplace_back(r->get_data_type());
    }
    dataType = PTO_TYPE::make_tuple(ret);
}

bool PTO_FOR_LOOP::type_check(STR_PTO_TYPE_MAP& validVar) {
    // 我们当前只支持两种for循环，range和parallel
    if (info->get_func_name() != "pypto.language.range" && info->get_func_name() != "pypto.language.parallel") {
        SPDLOG_ERROR("Unsupported for loop type '{}'", info->get_func_name());
        return false;
    }

    if (iter->type() != PTO_NODE_TYPE::VARIABLE) {
        SPDLOG_ERROR("Unexpected Error");
        return false;
    }
    
    if (validVar.find(iter->to_string()) != validVar.end()) {
        SPDLOG_WARN("Duplicated definition of '{}' at line {}", iter->to_string(), row_);
    }

    // 将iter的类型设为INT32
    iter->get_data_type().kind = PTO_TYPE_KIND::INT32;
    // 将iter存入table
    if (!add_var_to_map(iter, validVar)) return false;
    // validVar[iter->to_string()] = iter;

    // 需要拿到初始化列表
    std::vector<std::string> initList;
    auto args = info->get_arguments();
    for (const auto& arg : args) {
        if (arg->type() == PTO_NODE_TYPE::KEYWORD) {
            PTO_KEYWORD *ptr = (PTO_KEYWORD*)arg;
            if (ptr->get_keyword() != "init_values")
                continue;
            
            auto value = ptr->get_value();
            if (value->type() != PTO_NODE_TYPE::TUPLE_VARIABLE) {
                SPDLOG_INFO("Unexpected Error");
                return false;
            }

            const auto& varList = ((PTO_TUPLE_VAR*)value)->get_var_list();

            for (const auto& var : varList) {
                // 每一个var都应当是variable
                if (var->type() != PTO_NODE_TYPE::VARIABLE) {
                    SPDLOG_INFO("Unexpected Error");
                    return false;
                }
                initList.emplace_back(var->to_string());
            }
        }
    }

    // 应当存在于valid var中
    for (const auto& str : initList) {
        if (validVar.find(str) == validVar.end()) {
            SPDLOG_ERROR("Undefined variable '{}' at line {}", str, row_);
            return false;
        }
    }

    if (initList.size() != initVar.size()) {
        SPDLOG_ERROR("Mismatched length of init values at line {}", row_);
        return false;
    }

    for (std::size_t i = 0; i < initVar.size(); ++i) {
        initVar[i]->get_data_type() = validVar.find(initList[i])->second;
        if (!add_var_to_map(initVar[i], validVar)) 
            return false;
    }
    
    for (const auto& s : statements) {
        if (!s->type_check(validVar)) {
            return false;
        }
    }

    // 强制要求最后一个statement是yield
    if (statements.back()->type() != PTO_NODE_TYPE::ASSIGNMENT) {
        SPDLOG_ERROR("The last statement (line {}) of for loop is requried to be pypto.language.yield_",
            statements.back()->row()
        );
        return false;
    }
    auto ptr = (PTO_ASSIGNMENT*)statements.back();
    if (ptr->get_value()->type() != PTO_NODE_TYPE::FUNC_CALL) {
        SPDLOG_ERROR("The last statement (line {}) of for loop is requried to be pypto.language.yield_",
            statements.back()->row()
        );
        return false;
    }
    auto funcPtr = (PTO_CALL*)(ptr->get_value());
    if (funcPtr->get_func_name() != "pypto.language.yield_") {
        SPDLOG_ERROR("The last statement (line {}) of for loop is requried to be pypto.language.yield_",
            statements.back()->row()
        );
        return false;
    }

    if (ptr->get_lhs()->type() == PTO_NODE_TYPE::VARIABLE || ptr->get_lhs()->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
        if (initVar.size() != 1) {
            SPDLOG_ERROR("The number of variable in init list and yield is not matched at line {}: {} vs {}", row_, initVar.size(), 1);
            return false;
        }
        if (ptr->get_lhs()->get_data_type() != initVar[0]->get_data_type()) {
            SPDLOG_ERROR("Mismatched data type at line {} between {} and {}: {} vs {}",
                row_,
                ptr->get_lhs()->to_string(),
                initVar[0]->to_string(),
                ptr->get_lhs()->get_data_type().to_string(),
                initVar[0]->get_data_type().to_string()
            );
            return false;
        }
    }
    else if (ptr->get_lhs()->type() == PTO_NODE_TYPE::LIST_VARIABLE || ptr->get_lhs()->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
        std::vector<PTO_BASE*> varList;
        if (ptr->get_lhs()->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
            varList = ((PTO_LIST_VAR*)ptr->get_lhs())->get_var_list();
        } else {
            varList = ((PTO_TUPLE_VAR*)ptr->get_lhs())->get_var_list();
        }

        if (varList.size() != initVar.size()) {
            SPDLOG_ERROR("The number of variable in init list and yield is not matched at line {}: {} vs {}", row_, initVar.size(), varList.size());
            return false;
        }

        for (std::size_t i = 0; i < varList.size(); ++i) {
            if (varList[i]->get_data_type() != initVar[i]->get_data_type()) {
                SPDLOG_ERROR("Mismatched data type at line {} between {} and {}: {} vs {}",
                    row_,
                    varList[i]->to_string(),
                    initVar[i]->to_string(),
                    varList[i]->get_data_type().to_string(),
                    initVar[i]->get_data_type().to_string()
                );
                return false;
            }
        }
    }
    else {
        SPDLOG_ERROR("Unexpected Error");
        return false;
    }


    return true;
}

bool PTO_IF::type_check(STR_PTO_TYPE_MAP& validVar) {
    // Comparator的类型应当是bool
    comparator->infer_type(validVar);
    if (comparator->get_data_type().kind != PTO_TYPE_KIND::CONST_BOOL && comparator->get_data_type().kind != PTO_TYPE_KIND::BOOL) {
        SPDLOG_ERROR("Non-bool data type for if at line {}, {}", row_, comparator->get_data_type().to_string());
        return false;
    }

    for (const auto& ptr : ifStatement) {
        if (!ptr->type_check(validVar))
            return false;
    }

    for (const auto& ptr : elseStatement) {
        if (!ptr->type_check(validVar))
            return false;
    }

    return true;
}

bool PTO_FUNC::type_check(STR_PTO_TYPE_MAP& validVar) {
    // 函数名不应该重复
    if (validVar.find(funcName) != validVar.end()) {
        SPDLOG_ERROR("Duplicated definition of '{}'", funcName);
        return false;
    }

    // 推导arguments的类型
    for (const auto& ptr : arguments) {
        // 不处理self
        if (ptr->type() == PTO_NODE_TYPE::VARIABLE && ptr->to_string() == "self")
            continue;

        ptr->infer_type(validVar);

        // local变量名直接覆盖全局变量
        if (!add_var_to_map(ptr, validVar)) return false;
    }

    // 内部语句类型检查
    for (std::size_t i = 0; i < statements.size() - 1; ++ i) {
        if (!statements[i]->type_check(validVar)) return false;
    }

    // 最后一个语句应当是return
    if (statements.back()->type() != PTO_NODE_TYPE::RETURN) {
        SPDLOG_ERROR("The last statement of one function must be return");
        return false;
    }
    statements.back()->infer_type(validVar);

    // 基于returnTypeStr解析出这个函数的return类型
    if (returnTypeStr.size() == 0) {
        dataType = statements.back()->get_data_type();
    } else {
        std::vector<PTO_TYPE> temp;
        for (const auto& str : returnTypeStr) {
            if (str.find("pypto.language.Tensor") == 0) {
                // Tensor
                std::string content = str.substr(str.find_first_of('[') + 1);
                content.pop_back();

                // 第一个应当是dimension
                std::string dimension = content.substr(1, content.find_first_of(']') - 1);
                content = content.substr(content.find_first_of(']') + 2);

                std::vector<int> shape;
                while (true) {
                    size_t index = dimension.find_first_of(',');
                    if (index == std::string::npos) {
                        shape.emplace_back(std::stoi(dimension));
                        break;
                    } else {
                        shape.emplace_back(std::stoi(dimension.substr(0, index)));
                        dimension = dimension.substr(index + 1);
                    }
                }
                // 第二个是数据类型
                std::string tensorType = content;
                if (content.find_first_of(',') == std::string::npos) {
                    content = "";
                } else {
                    tensorType = content.substr(0, content.find_first_of(','));
                    content = content.substr(content.find_first_of(',') + 1);
                }

                if (tensorType == "pypto.language.BF16") {
                    temp.emplace_back(PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::BF16));
                }
                else if (tensorType == "pypto.language.FP32") {
                    temp.emplace_back(PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::FP32));
                }
                else if (tensorType == "pypto.language.INT32") {
                    temp.emplace_back(PTO_TYPE::make_tensor(shape, PTO_TYPE_KIND::INT32));
                }
                else {
                    SPDLOG_ERROR("Unimplemented data type '{}'", tensorType);
                }

                if (content != "") {
                    SPDLOG_ERROR("Unprocessed type '{}'", content);
                }
            }
            else if (str == "pypto.language.Scalar[pypto.language.INDEX]") {
                temp.emplace_back(PTO_TYPE::make_scalar(PTO_TYPE_KIND::INT32));
            }
            else {
                SPDLOG_ERROR("Unprocessed data type {}", str);
            }
        }
        
        this->dataType = PTO_TYPE::make_tuple(temp);
    }

    if (this->dataType != statements.back()->get_data_type()) {
        SPDLOG_ERROR("Mismatched return type for func {} at line {}, {} vs {}",
            funcName,
            row_,
            this->dataType.to_string(),
            statements.back()->get_data_type().to_string()
        );
    }

    return true;
}

bool PTO_CLASS::type_check(STR_PTO_TYPE_MAP& validVar) {
    if (validVar.find(name) != validVar.end()) {
        SPDLOG_ERROR("Duplicated definition of '{}'", name);
        return false;
    }
    validVar[name] = this->dataType;

    for (const auto& ptr : functions) {
        // 复制一份变量列表，进入class内部
        STR_PTO_TYPE_MAP localValidVar = validVar;
        if (!ptr->type_check(localValidVar)) {
            return false;
        }

        // 将函数定义放到valid var里
        if (validVar.find(ptr->get_func_name()) != validVar.end()) {
            SPDLOG_ERROR("Duplicated definition for {}", ptr->get_func_name());
        }
        validVar[ptr->get_func_name()] = ptr->get_data_type();
    }
    
    return true;
}

bool PTO_MODULE::type_check() const {
    // 顺序扫描AST，完成类型检查

    // 用于记录扫描过的变量
    STR_PTO_TYPE_MAP validVar;
    // 用于记录扫描过的函数和class

    // 先扫描全局变量
    for (const auto& it : globalVariable) {
        if (!it.second->type_check(validVar)) {
            return false;
        }
    }

    // 再扫描class或func
    for (const auto& it : classOrFunc) {
        if (!it->type_check(validVar)) {
            return false;
        }
    }

    return true;
}

}