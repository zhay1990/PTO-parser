#include "ptoNode.hh"
#include "logger.hh"
#include <sstream>

namespace pto_parser
{
// 用于记录program ID信息
struct PROGRAM_ID{
    int axis; // grid的第几个维度？
    int range; // kernel函数内部的循环范围，对所有kernel函数的调用是相同的值；对于host内产生的programID，则为-1
    std::unordered_map<PTO_CALL*, int> rangePerCall; // Host内每次调用kernel函数可能有不同的循环次数，在此处记录

    PROGRAM_ID()
        : axis(-1),
          range(-1),
          rangePerCall()
    {}
};

// 记录每个函数名，有哪些变量是循环变量
// 第一个index是kernel函数的名字，第二个index是入参的序号
static std::unordered_map<std::string, std::unordered_map<int, struct PROGRAM_ID>> programIDInfo;


// 入参里的tensor都是deviceMemory的指针
static std::unordered_map<std::string, std::vector<std::string>> deviceMemoryPtr;
// 记录当前kernel函数内哪些devicememory指针被直接使用
static std::unordered_set<std::string> directUsedDeviceMemory;
// 记录每个kernel函数的第几个output指针需要新建
static std::unordered_map<std::string, std::unordered_set<int>> kernelOutputPtr;
// 记录当前kernel函数的返回变量名
static std::vector<std::string> curOutputName;
// 转换成triton的过程中，不是一对一转化，可能需要生成新的中间变量，所以这里记录kernenl函数内部用到的变量名，避免错误
static std::unordered_set<std::string> kernelUsedVarName;



static void generate_offset(const std::string& indent, std::ofstream& fout, const std::string& lhsName, const std::string& tensorName, PTO_LIST_VAR* subShape, PTO_LIST_VAR* startPos) {
    if (subShape->get_var_list()[0]->type() == PTO_NODE_TYPE::INT_CONSTANT && subShape->get_var_list()[1]->type() == PTO_NODE_TYPE::INT_CONSTANT) {
        fout << indent << lhsName << "_offset = " << 
        "(" << startPos->get_var_list()[0]->to_string() << " + tl.arange(0, " << subShape->get_var_list()[0]->to_string() << "))[:, None] * " << tensorName << "_stride_0 + " <<
        "(" << startPos->get_var_list()[1]->to_string() << " + tl.arange(0, " << subShape->get_var_list()[1]->to_string() << "))[None, :] * " << tensorName << "_stride_1" << std::endl;
    }
    else {
        SPDLOG_ERROR("Non-constant subshape");
    }
}

void PTO_MODULE::convert_to_triton(const std::string& fileName) const {
    // 简化处理，假设一个文件只有一个class
    if (classOrFunc.size() != 1 || classOrFunc[0]->type() != PTO_NODE_TYPE::CLASS) {
        SPDLOG_ERROR("Only support one class in one file");
        return;
    }

    // 先写入头文件引用等基本信息
    std::ofstream fout(fileName, std::ios::out);

    if (!fout.is_open()) {
        SPDLOG_ERROR("Unable to open {}", fileName);
        return;
    }

    fout << "import torch" << std::endl
         << "import triton.language as tl" << std::endl
         << "import triton" << std::endl << std::endl;

    
    // 将内容转换成triton代码
    ((PTO_CLASS*)classOrFunc[0])->convert_to_triton(fout);

    fout.close();
}

void PTO_CLASS::convert_to_triton(std::ofstream& fout) const {
    // 输出class定义
    fout << "class " << name << "_triton:" << std::endl;


    // 寻找到host函数，现在强制要求只有一个host函数
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

    SPDLOG_DEBUG("Found host function: {}", ptr->get_func_name());

    // 扫描host函数，确认每个triton kernel函数的哪个入参对应program id，具体判断标准如下
    // 如果kernel函数是一个parallel loop的唯一statement，则该parallel loop的iterator就是通过program ID传进内核函数
    // 扫描过程中会删除parallel循环，简化后续代码的生成
    ptr->triton_kernel_call_conversion();

    // 带着PROGRAM ID信息处理kernel函数，并同时生成对应的triton代码
    for (const auto& func : functions) {
        func->convert_to_triton_kernel(1, fout);
    }

    // ptr->convert_to_triton_host(1, fout);
}

PTO_BASE* PTO_FUNC::triton_kernel_call_conversion() {
    for (auto& ptr : statements) {
        auto newPtr = ptr->triton_kernel_call_conversion();
        
        if (newPtr == nullptr) {
            continue;
        }

        // 如果newPtr不是null，则表明这个statement是一个parallel loop+kernel函数调用
        delete ptr;
        ptr = newPtr;
    }

    return nullptr;
}

PTO_BASE* PTO_IF::triton_kernel_call_conversion() {
    for (auto& ptr : ifStatement) {
        auto newPtr = ptr->triton_kernel_call_conversion();

        if (newPtr == nullptr) {
            continue;
        }

        delete ptr;
        ptr = newPtr;
    }
    for (auto& ptr : elseStatement) {
        auto newPtr = ptr->triton_kernel_call_conversion();

        if (newPtr == nullptr) {
            continue;
        }

        delete ptr;
        ptr = newPtr;
    }
    return nullptr;
}

PTO_BASE* PTO_FOR_LOOP::triton_kernel_call_conversion() {
    // 可能出现parallel循环嵌套的情况，所以先处理内部的statement
    for (auto& ptr : statements) {
        auto newPtr = ptr->triton_kernel_call_conversion();

        if (newPtr == nullptr) {
            continue;
        }

        delete ptr;
        ptr = newPtr;
    }

    // 判断该循环是否是一个只包含kernel函数调用的paralle循环
    if (info->get_func_name() != "pypto.language.parallel") {
        return nullptr;
    }

    if (statements.size() != 1) {
        return nullptr;
    }

    if (statements[0]->type() != PTO_NODE_TYPE::ASSIGNMENT) {
        return nullptr;
    }

    auto assignPtr = (PTO_ASSIGNMENT*)statements[0];

    if (assignPtr->get_value()->type() != PTO_NODE_TYPE::FUNC_CALL) {
        return nullptr;
    }

    auto funcPtr = (PTO_CALL*)assignPtr->get_value();

    if (funcPtr->get_func_name().substr(0, 5) != "self.") {
        // 这里依旧假定其他函数都是kernel函数
        return nullptr;
    }

    auto kernelName = funcPtr->get_func_name().substr(5);

    // 循环变量是program ID，先确认对应的函数入参是第几个
    int index = -1;
    for (std::size_t i = 0; i < funcPtr->get_arguments().size(); ++i) {
        const auto& arg = funcPtr->get_arguments()[i];

        if (arg->type() == PTO_NODE_TYPE::VARIABLE) {
            if (arg->to_string() == iter->to_string()) {
                if (index != -1) {
                    // 一个parallel loop控制两个维度？？
                    SPDLOG_ERROR("Unexpected Error");
                    return nullptr;
                }
                index = (int)i;
            }
        }
        else {
            SPDLOG_ERROR("Unimplemented");
            return nullptr;
        }
    }

    if (index == -1) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }

    // 登记这个program ID的信息
    if (programIDInfo.find(kernelName) == programIDInfo.end()) {
        programIDInfo[kernelName] = std::unordered_map<int, struct PROGRAM_ID>();
    }
    if (programIDInfo[kernelName].find(index) == programIDInfo[kernelName].end()) {
        programIDInfo[kernelName][index] = PROGRAM_ID();
        programIDInfo[kernelName][index].axis = programIDInfo[kernelName].size() - 1;
        programIDInfo[kernelName][index].range = -1;
    }

    if (programIDInfo[kernelName][index].range != -1) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }

    // 登记这次循环的range
    if (info->get_arguments()[0]->type() != PTO_NODE_TYPE::INT_CONSTANT) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }
    programIDInfo[kernelName][index].rangePerCall[(PTO_CALL*)assignPtr->get_value()] = std::stoi(info->get_arguments()[0]->to_string());

    // 为删除这个parallel loop节点做准备
    statements.clear();

    // 返回这个assign指针
    return assignPtr;
}

void PTO_FUNC::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // 只处理内核函数
    if (decorate != "pypto.language.function(type = pypto.language.FunctionType.InCore)") {
        return;
    }

    // 生成函数定义
    std::string indent(depth, '\t');
    fout << indent << "@staticmethod" << std::endl;
    fout << indent << "@triton.jit" << std::endl;
    fout << indent << "def " << funcName << "(" << std::endl;

    auto& curProgramIDInfo = programIDInfo[funcName];
    deviceMemoryPtr.clear();
    directUsedDeviceMemory.clear();
    curOutputName.clear();
    kernelUsedVarName.clear();

    std::vector<std::string> programIDName(curProgramIDInfo.size(), "");

    // 先记录deviceMemory的名字
    // 默认第一个入参是self
    if (arguments[0]->to_string() != "self") {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    for (std::size_t i = 1; i < arguments.size(); ++i) {
        const auto& argType = arguments[i]->get_data_type();

        if (argType.kind == PTO_TYPE_KIND::UNKNOWN) {
            SPDLOG_ERROR("Should call type_check() before converting to triton");
            return;
        }

        if (argType.kind == PTO_TYPE_KIND::TENSOR) {
            // Tensor不可能存在于program id中
            if (curProgramIDInfo.find(i - 1) != curProgramIDInfo.end()) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            // 简化处理，强制要求均为二维tensor
            if (argType.shape.size() != 2) {
                SPDLOG_ERROR("Only support two-dimensional tensor");
                return;
            }

            // 先记录这个变量名字，等扫描完成再确定生成的名字
            deviceMemoryPtr[arguments[i]->to_string()] = std::vector<std::string>();
        }
    }

    // 扫描所有statement，完成三项任务
    // 1. 收集使用过的变量名
    // 2. 收集被直接使用的deviceMemory的名字
    // 3. 收集返回值的名字，判断是否需要增加入参
    for (const auto& s : statements) {
        s->collect_triton_kernel_info();
    }


    // 入参里的tensor都处理成device memory指针，生成三个对应入参，xx_ptr, xx_stride_0, xx_stride_1，后两个是用于计算offset
    // 如果出现重名，则在xx_ptr后面加_
    for (std::size_t i = 1; i < arguments.size(); ++i) {
        const auto& argType = arguments[i]->get_data_type();

        if (argType.kind == PTO_TYPE_KIND::TENSOR) {
            auto& convertName = deviceMemoryPtr[arguments[i]->to_string()];
            std::string name = arguments[i]->to_string() + "_ptr";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            name = arguments[i]->to_string() + "_stride_0";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            name = arguments[i]->to_string() + "_stride_1";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            fout << indent << "\t#" << arguments[i]->get_type_str()[0] << std::endl;
            fout << indent << "\t" << convertName[0] << ", " << convertName[1] << ", " << convertName[2] << std::endl;
        }
        else if (argType.kind == PTO_TYPE_KIND::INT32) {
            kernelUsedVarName.insert(arguments[i]->to_string());
            if (curProgramIDInfo.find(i - 1) == curProgramIDInfo.end()) {
                fout << indent << "\t" << arguments[i]->to_string() << "," << std::endl;
            }
            else {
                programIDName[curProgramIDInfo[i - 1].axis] = arguments[i]->to_string();
            }
        }
        else {
            SPDLOG_ERROR("Unexpected argument type {}", argType.to_string());
            return;
        }
    }

    // 处理输出
    for (std::size_t i = 0; i < curOutputName.size(); ++i) {
        if (deviceMemoryPtr.find(curOutputName[i]) == deviceMemoryPtr.end()) {
            // 这个输出变量不在入参中，需要添加
            kernelOutputPtr[funcName].insert(i);

            std::vector<std::string> convertName;

            std::string name = curOutputName[i] + "_ptr";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            name = curOutputName[i] + "_stride_0";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            name = curOutputName[i] + "_stride_1";
            while (kernelUsedVarName.find(name) != kernelUsedVarName.end()) {
                name += '_';
            }
            kernelUsedVarName.insert(name);
            convertName.emplace_back(name);

            deviceMemoryPtr[curOutputName[i]] = convertName;

            fout << indent << "\t#" << returnTypeStr[i] << std::endl;
            fout << indent << "\t" << convertName[0] << ", " << convertName[1] << ", " << convertName[2] << ", # Output ptr" << std::endl;
        }
    }
    
    fout << indent << "):" << std::endl;


    // 如果这个函数只有两个statement，一个是paralle loop，一个是return
    // 则parallel loop处理成grid的一个维度
    // 其他parallel loop都当成是串行的range处理
    int programIDIndex = -1;
    while (statements.size() == 2 && statements[0]->type() == PTO_NODE_TYPE::FOR_LOOP && statements[1]->type() == PTO_NODE_TYPE::RETURN) {
        auto forPtr = (PTO_FOR_LOOP*)statements[0];
        auto newStatements = forPtr->extract_kernel_parallelism();
        if (newStatements.size() == 0) {
            break;
        }

        // 记录新的program ID
        // index用负数表示
        programIDName.emplace_back(forPtr->get_iter()->to_string());
        curProgramIDInfo[programIDIndex] = PROGRAM_ID();
        curProgramIDInfo[programIDIndex].axis = curProgramIDInfo.size() - 1;
        curProgramIDInfo[programIDIndex].range = std::stoi(forPtr->get_info()->get_arguments()[0]->to_string());
        programIDIndex -= 1;

        delete statements[0];
        newStatements.emplace_back(statements[1]);
        this->statements = newStatements;
    }

    // 先输出programID
    for (std::size_t i = 0; i < programIDName.size(); ++i) {
        fout << indent << "\t" << programIDName[i] << " = tl.program_id(axis = " << i << ")" << std::endl;
    }

    // 输出使用到的deviceMemory
    for (const auto& var : directUsedDeviceMemory) {
        fout << indent << "\t" << var << " = tl.load(" << deviceMemoryPtr[var][0] << ")" << std::endl;
    }

    // if (statements.size() == 2 && 
    //     statements[0]->type() == PTO_NODE_TYPE::FOR_LOOP && 
    //     ((PTO_FOR_LOOP*)statements[0])->get_info()->type() == PTO_NODE_TYPE::FUNC_CALL &&
    //     ((PTO_CALL*)((PTO_FOR_LOOP*)statements[0])->get_info())->get_func_name() == "pypto.language.parallel" &&
    //     statements[1]->type() == PTO_NODE_TYPE::RETURN
    // ) {
    //     // 处理program ID, parallel的iter是个新的program_ID
    //     auto forPtr = (PTO_FOR_LOOP*)statements[0];

    //     curProgramIDInfo

    //     programIDInfo[forPtr->get_iter()->to_string()].axis = programIDInfo.size();
    //     programIDInfo[forPtr->get_iter()->to_string()].range = std::stoi(forPtr->get_info()->get_arguments()[0]->to_string());

    //     for (const auto& it : programIDInfo) {
    //         fout << indent << "\t" << it.first << " = tl.program_id(axis = " << it.second.axis << ")" << std::endl;
    //     }

    //     // 先扫描所有statement，是否有直接使用input tensor的？需要在这里先增加对应的tl.load指令
    //     // 注意，这里不需要考虑slice，assemble函数需要特殊考虑

    //     for (const auto& s: forPtr->get_statements()) {
    //         s->convert_to_triton_kernel(depth + 1, fout);
    //         fout << std::endl;
    //     }
    // } else {
    //     // 先扫描所有statement，是否有直接使用input tensor的？需要在这里先增加对应的tl.load指令
    //     // 注意，这里不需要考虑slice，assemble函数需要特殊考虑

    //     for (const auto& it : statements) {
    //         it->convert_to_triton_kernel(depth + 1, fout);
    //         fout << std::endl;
    //     }
    // }

    // fout << std::endl;
}

void PTO_ASSIGNMENT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');

    // // 如果右侧是funcCall，有几个情况需要特殊处理
    // if (value->type() == PTO_NODE_TYPE::FUNC_CALL) {
    //     auto ptr = (PTO_CALL*)value;
    //     if (ptr->get_func_name() == "pypto.language.tensor.slice") {
    //         const auto tensorName = ptr->get_arguments()[0]->to_string();

    //         // 强制要求是二维tensor
    //         const auto subShape = (PTO_LIST_VAR*)ptr->get_arguments()[1];
    //         const auto startPos = (PTO_LIST_VAR*)ptr->get_arguments()[2];

    //         if (subShape->get_var_list().size() != 2) {
    //             SPDLOG_ERROR("Only support two-dimensional tensor");
    //             return;
    //         }

    //         if (deviceMemoryPtr.find(tensorName) != deviceMemoryPtr.end()) {
    //             generate_offset(indent, fout, lhs->to_string(), tensorName, subShape, startPos);
    //             fout << indent << lhs->to_string() << " = tl.load(" << tensorName << "_ptr + " << lhs->to_string() << "_offset)";
    //         }
    //         else {
    //             fout << indent << lhs->to_string() << " = " << tensorName << "[" <<
    //                             startPos->get_var_list()[0]->to_string() << ": " << startPos->get_var_list()[0]->to_string() << " + " << subShape->get_var_list()[0]->to_string() << ", " <<
    //                             startPos->get_var_list()[1]->to_string() << ": " << startPos->get_var_list()[1]->to_string() << " + " << subShape->get_var_list()[1]->to_string() << "]";
    //         }

    //         return;
    //     }
    //     else if (ptr->get_func_name() == "pypto.language.tensor.assemble") {
    //         if (deviceMemoryPtr.find(lhs->to_string()) != deviceMemoryPtr.end()) {
    //             if (lhs->to_string() != ptr->get_arguments()[0]->to_string()) {
    //                 // 要加一个store语句
    //                 fout << indent << "tl.store(" << lhs->to_string() << "_ptr, " << ptr->get_arguments()[0]->to_string() << ")" << std::endl;
    //             }

    //             // 拿到subshape
    //             const auto& argDataType = ptr->get_arguments()[1]->get_data_type();
    //             if (argDataType.kind != PTO_TYPE_KIND::TENSOR || argDataType.shape.size() != 2) {
    //                 SPDLOG_ERROR("Expect two-dimensional tensor");
    //                 return;
    //             }
    //             fout << indent << lhs->to_string() << "_offset = " << 
    //                 "(" << ((PTO_LIST_VAR*)ptr->get_arguments()[2])->get_var_list()[0]->to_string() << " + tl.arange(0, " << argDataType.shape[0] << "))[:, None] * " << lhs->to_string() << "_stride_0 + " <<
    //                 "(" << ((PTO_LIST_VAR*)ptr->get_arguments()[2])->get_var_list()[1]->to_string() << " + tl.arange(0, " << argDataType.shape[1] << "))[None, :] * " << lhs->to_string() << "_stride_1" << std::endl;
    //             fout << indent << "tl.store(" << lhs->to_string() << "_ptr + " << lhs->to_string() << "_offset, " << ptr->get_arguments()[1]->to_string() << ")";
    //         }
    //         else {
    //             // 做一个强限制，要求某个维度必须是1
    //             const auto& lhsDataType = lhs->get_data_type();
    //             if (lhsDataType.kind != PTO_TYPE_KIND::TENSOR || lhsDataType.shape.size() != 2) {
    //                 SPDLOG_ERROR("Only support two-dimension tensor");
    //                 return;
    //             }
    //             if (lhsDataType.shape[0] != 1 && lhsDataType.shape[1] != 1) {
    //                 SPDLOG_ERROR("One dimension must be 1");
    //                 return;
    //             }

    //             // 先做一个赋值
    //             fout << indent << lhs->to_string() << " = " << ptr->get_arguments()[0]->to_string() << std::endl;

    //             // 用tl.reshape展平成一维

                
    //             SPDLOG_ERROR("Assemble to non-device ptr");
    //         }
    //         return;
    //     }
    // }

    // fout << indent << lhs->to_string() << " = ";
    // value->convert_to_triton_kernel(0, fout);

}

void PTO_CALL::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // fout << indent;
    // if (funcName == "pypto.language.tensor.create") {
    //     fout << "tl.zeros(";
    //     for (std::size_t i = 0; i < arguments.size(); ++i) {
    //         arguments[i]->convert_to_triton_kernel(0, fout);
    //         if (i != arguments.size() - 1) {
    //             fout << ", ";
    //         }
    //     }
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.tensor.muls" || funcName == "pypto.language.tensor.mul" || funcName == "pypto.language.tensor.row_expand_mul" || funcName == "pypto.language.tensor.col_expand_mul") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << " * ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    // }
    // else if (funcName == "pypto.language.tensor.cast" || funcName == "pypto.language.cast") {
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ".to(";
    //     if (arguments[1]->type() == PTO_NODE_TYPE::KEYWORD) {
    //         arguments[1]->convert_to_triton_kernel(0, fout);
    //     }
    //     else if (arguments[1]->to_string() == "pypto.language.INDEX") {
    //         fout << "tl.int32";
    //     }
    //     else {
    //         SPDLOG_ERROR("Unimplemented arg {}", arguments[1]->to_string());
    //     }
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.tensor.row_sum") {
    //     fout << "tl.sum(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ", axis = 1)[:, None]";
    // }
    // else if (funcName == "pypto.language.tensor.adds" || funcName == "pypto.language.tensor.add") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << " + ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    // }
    // else if (funcName == "pypto.language.tensor.sub" || funcName == "pypto.language.tensor.row_expand_sub") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << " - ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    // }
    // else if (funcName == "pypto.language.tensor.row_expand_div") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << " / ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    // }
    // else if (funcName == "pypto.language.tensor.rsqrt") {
    //     fout << "tl.math.rsqrt(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.tensor.exp") {
    //     fout << "tl.exp(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.tensor.matmul") {
    //     fout << "tl.dot(";
    //     // a_trans是否为真
    //     bool a_trans = false, b_trans = false;
    //     for (const auto& arg : arguments) {
    //         if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "a_trans") {
    //             if (((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
    //                 a_trans = true;
    //             }
    //         }
    //         if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "b_trans") {
    //             if (((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
    //                 b_trans = true;
    //             }
    //         }
    //     }

    //     if (a_trans) {
    //         fout << "tl.trans(";
    //         arguments[0]->convert_to_triton_kernel(0, fout);
    //         fout << "), ";
    //     } else {
    //         arguments[0]->convert_to_triton_kernel(0, fout);
    //         fout << ", ";
    //     }
    //     if (b_trans) {
    //         fout << "tl.trans(";
    //         arguments[1]->convert_to_triton_kernel(0, fout);
    //         fout <<")";
    //     } else {
    //         arguments[1]->convert_to_triton_kernel(0, fout);
    //     }
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.tensor.row_max") {
    //     fout << "tl.max(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ", axis = 1)[:, None]";
    // }
    // else if (funcName == "pypto.language.tensor.maximum") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     fout << "tl.maximum(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ", ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    //     fout << ")";
    // }
    // else if (funcName == "pypto.language.min") {
    //     if (arguments.size() != 2) {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }

    //     fout << "tl.minimum(";
    //     arguments[0]->convert_to_triton_kernel(0, fout);
    //     fout << ", ";
    //     arguments[1]->convert_to_triton_kernel(0, fout);
    //     fout << ")";
    // }
    // else {
    //     SPDLOG_ERROR("Unimplemented function name {}", funcName);
    // }

}

void PTO_TUPLE_VAR::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // fout << indent << "(";
    // for (std::size_t i = 0; i < varList.size(); ++i) {
    //     varList[i]->convert_to_triton_kernel(0, fout);
    //     if (i != varList.size() - 1) {
    //         fout << ", ";
    //     }
    // }
    // fout << ")";
}

void PTO_LIST_VAR::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // fout << indent << "[";
    // for (std::size_t i = 0; i < varList.size(); ++i) {
    //     varList[i]->convert_to_triton_kernel(0, fout);
    //     if (i != varList.size() - 1) {
    //         fout << ", ";
    //     }
    // }
    // fout << "]";
}

void PTO_KEYWORD::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // if (keyword == "dtype") {
    //     fout << indent << keyword << " = ";

    //     std::string temp = value->to_string();

    //     if (temp == "pypto.language.FP32") {
    //         fout << "tl.float32";
    //     }
    //     else if (temp == "pypto.language.BF16") {
    //         fout << "tl.bfloat16";
    //     }
    //     else {
    //         SPDLOG_ERROR("Unimplemented data type {}", temp);
    //     }
    // }
    // else if (keyword == "layout") {
    //     // 忽略
    // }
    // else if (keyword == "target_type") {
    //     std::string temp = value->to_string();
    //     if (temp == "pypto.language.FP32") {
    //         fout << "tl.float32";
    //     }
    //     else if (temp == "pypto.language.BF16") {
    //         fout << "tl.bfloat16";
    //     }
    //     else {
    //         SPDLOG_ERROR("Unimplemented data type {}", temp);
    //     }
    // }
    // else {
    //     SPDLOG_ERROR("Unimplemented keyword {}" , keyword);
    // }
}

void PTO_INT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // fout << std::string(depth, '\t') << value;
}

void PTO_FLOAT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // fout << std::string(depth, '\t') << value;
}

void PTO_BOOL::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // if (value) {
    //     fout << std::string(depth, '\t') << "True";
    // }
    // else {
    //     fout << std::string(depth, '\t') << "False";
    // }
}

void PTO_VARIABLE::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // fout << std::string(depth, '\t') << varName;
}

void PTO_FOR_LOOP::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // 处理成串行的range
    // std::string indent(depth, '\t');
    // fout << indent << "for ";
    // iter->convert_to_triton_kernel(0, fout);
    // fout << " in range(";
    // info->get_arguments()[0]->convert_to_triton_kernel(0, fout);
    // fout << "):" << std::endl;
    // for (const auto& s : statements) {
    //     s->convert_to_triton_kernel(depth + 1, fout);
    //     fout << std::endl;
    // }
}

void PTO_BINARY_OP::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // fout << indent << "(";
    // lhs->convert_to_triton_kernel(0, fout);

    // switch (op) {
    //     case PTO_OPERATOR::ADD: fout << " + "; break;
    //     case PTO_OPERATOR::SUB: fout << " - "; break;
    //     case PTO_OPERATOR::FLOOR_DIV: fout << " // "; break;
    //     case PTO_OPERATOR::MUL: fout << " * "; break;
    //     case PTO_OPERATOR::EQUAL: fout << " == "; break;
    // }
    // rhs->convert_to_triton_kernel(0, fout);
    // fout << ")";
}

void PTO_RETURN::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // for (const auto& val : returnVal) {
    //     fout << indent << "tl.store(" << val->to_string() << "_ptr, " << val->to_string() << ")";
    // }

}

void PTO_IF::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // std::string indent(depth, '\t');
    // fout << indent << "if ";
    // comparator->convert_to_triton_kernel(0, fout);
    // fout << ":" << std::endl;
    // for (const auto& s : ifStatement) {
    //     s->convert_to_triton_kernel(depth + 1, fout);
    //     fout << std::endl;
    // }
    // fout << indent << "else:" << std::endl;
    // for (const auto& s : elseStatement) {
    //     s->convert_to_triton_kernel(depth + 1, fout);
    //     fout << std::endl;
    // }
}

void PTO_VARIABLE::collect_triton_kernel_info() const {
    kernelUsedVarName.insert(varName);
}

void PTO_TUPLE_VAR::collect_triton_kernel_info() const {
    for (const auto& var : varList) {
        var->collect_triton_kernel_info();
    }
}

void PTO_LIST_VAR::collect_triton_kernel_info() const {
    for (const auto& var : varList) {
        var->collect_triton_kernel_info();
    }
}

void PTO_BINARY_OP::collect_triton_kernel_info() const {
    // 收集使用过的变量名
    lhs->collect_triton_kernel_info();
    rhs->collect_triton_kernel_info();

    // 是否有直接使用的device memory变量
    if (lhs->type() == PTO_NODE_TYPE::VARIABLE) {
        if (deviceMemoryPtr.find(lhs->to_string()) != deviceMemoryPtr.end()) {
            directUsedDeviceMemory.insert(lhs->to_string());
        }
    }
    if (rhs->type() == PTO_NODE_TYPE::VARIABLE) {
        if (deviceMemoryPtr.find(rhs->to_string()) != deviceMemoryPtr.end()) {
            directUsedDeviceMemory.insert(rhs->to_string());
        }
    }
}

void PTO_CALL::collect_triton_kernel_info() const {
    // 收集使用过的变量名
    for (const auto& arg : arguments) {
        arg->collect_triton_kernel_info();
    }

    // 是否有直接使用的device memory变量
    // 这里需要根据函数的类型确认
    if (funcName == "pypto.language.tensor.assemble") {
        // 这里只能先判断第二个变量是否是device memory
        if (deviceMemoryPtr.find(arguments[1]->to_string()) != deviceMemoryPtr.end()) {
            // 直接使用
            directUsedDeviceMemory.insert(arguments[1]->to_string());
        }
    }
    else if (funcName == "pypto.language.tensor.slice") {
        // 无需检查
    }
    else {
        for (const auto& arg : arguments) {
            if (arg->type() == PTO_NODE_TYPE::VARIABLE && deviceMemoryPtr.find(arg->to_string()) != deviceMemoryPtr.end()) {
                directUsedDeviceMemory.insert(arg->to_string());
            }
        }
    }
}

void PTO_ASSIGNMENT::collect_triton_kernel_info() const {
    // 收集使用过的变量名
    lhs->collect_triton_kernel_info();
    value->collect_triton_kernel_info();

    // 特殊情况处理
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL) {
        auto funcPtr = (PTO_CALL*)value;

        if (funcPtr->get_func_name() == "pypto.language.tensor.assemble") {
            // 如果第一个argument和lhs名字不一样，且是device memory，则是direct access
            if (funcPtr->get_arguments()[0]->to_string() != lhs->to_string() && deviceMemoryPtr.find(funcPtr->get_arguments()[0]->to_string()) != deviceMemoryPtr.end()) {
                directUsedDeviceMemory.insert(funcPtr->get_arguments()[0]->to_string());
            }
        }
    }
}

void PTO_RETURN::collect_triton_kernel_info() const {
    for (const auto& val : returnVal) {
        val->collect_triton_kernel_info();
    }

    // 做一个强制要求，如果kernel有多个return statement，则每个statement返回的值必须一样
    if (curOutputName.size() != 0) {
        if (curOutputName.size() != returnVal.size()) {
            SPDLOG_ERROR("Mismatched return variable name");
            return;
        }
        for (std::size_t i = 0; i < curOutputName.size(); ++i) {
            if (curOutputName[i] != returnVal[i]->to_string()) {
                SPDLOG_ERROR("Mismatched return variable name");
                return;
            }
        }
    } else {
        for (const auto& val : returnVal) {
            // 强制要求是二维tile
            if (val->get_data_type().kind != PTO_TYPE_KIND::TENSOR || val->get_data_type().shape.size() != 2) {
                SPDLOG_ERROR("Only support two-dimensional tensor");
                return;
            }
            curOutputName.emplace_back(val->to_string());
        }
    }
}

void PTO_FOR_LOOP::collect_triton_kernel_info() const {
    for (const auto& s : statements) {
        s->collect_triton_kernel_info();
    }
}

void PTO_IF::collect_triton_kernel_info() const {
    for (const auto& s : ifStatement) {
        s->collect_triton_kernel_info();
    }
    for (const auto& s : elseStatement) {
        s->collect_triton_kernel_info();
    }
}

std::vector<PTO_BASE*> PTO_FOR_LOOP::extract_kernel_parallelism() {
    std::vector<PTO_BASE*> ret;
    if (info->get_func_name() == "pypto.language.parallel") {
        ret = statements;
        statements.clear();
    }

    return ret;
}

static void create_device_memory_on_host(const std::string& indent, std::ofstream& fout, const std::string& name, const std::vector<int>& shape, const std::string& type) {
    // if (deviceMemoryPtr.find(name) != deviceMemoryPtr.end()) return;
    // deviceMemoryPtr.insert(name);

    // if (shape.size() != 2) {
    //     SPDLOG_ERROR("Only support two-dimensional tensor");
    //     return;
    // }

    // fout << indent << name << " = torch.empty([" << shape[0] << ", " << shape[1] << "], " <<
    //                           "dtype = torch." << type << ", device = 'cuda')" << std::endl;
}

void PTO_FUNC::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // std::string indent(depth, '\t');

    // // 记录哪些变量的device memory已经创建，后续不会在重复创建
    // deviceMemoryPtr.clear();

    // fout << indent << "def " << funcName << "(";
    // for (const auto& arg : arguments) {
    //     fout << arg->to_string() << ", ";

    //     if (arg->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
    //         deviceMemoryPtr.insert(arg->to_string());
    //     }
    // }
    // fout << "):" << std::endl;


    // for (const auto& s : statements) {
    //     s->convert_to_triton_host(depth + 1, fout);
    //     fout << std::endl;
    // }
}

void PTO_ASSIGNMENT::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // std::string indent(depth, '\t');

    // // 对于kernel函数的调用需要特殊处理
    // if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name().substr(0, 5) == "self.") {
    //     // 先要创建output memory
    //     const auto& lhsDataType = lhs->get_data_type();
    //     if (lhsDataType.kind == PTO_TYPE_KIND::TENSOR) {
    //         if (lhsDataType.shape.size() != 2) {
    //             SPDLOG_ERROR("Only support two-dimensional tensor");
    //             return;
    //         }
    //         std::string tensorType = "";
    //         if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::BF16) {
    //             tensorType = "bfloat16";
    //         }
    //         else if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::FP32) {
    //             tensorType = "float32";
    //         }
    //         else {
    //             SPDLOG_ERROR("Unsupported data type");
    //             return;
    //         }
    //         create_device_memory_on_host(indent, fout, lhs->to_string(), lhsDataType.shape, tensorType);

    //         value->convert_to_triton_host(depth, fout);

    //         // 处理输出
    //         fout << lhs->to_string() << ", " << lhs->to_string() << ".stride(0), " << lhs->to_string() << ".stride(1))";
    //     }
    //     else {
    //         SPDLOG_ERROR("Unimplemented");
    //         return;
    //     }
    // }
    // // 对于assemble需要特殊处理
    // else if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.tensor.assemble") {
    //     // 这个函数特殊处理，如果第一个argument和lhs不是同一个变量，则需要拆成两个assignment
    //     const auto& args = ((PTO_CALL*)value)->get_arguments();

    //     if (args[0]->to_string() != lhs->to_string()) {
    //         lhs->dump_to_pyTorch(depth, fout);
    //         fout << " = ";
    //         args[0]->dump_to_pyTorch(0, fout);
    //         fout << std::endl;
    //     }
        
    //     lhs->dump_to_pyTorch(depth, fout);
    //     fout << " [";

    //     std::vector<PTO_BASE*> varList;
    //     if (args[2]->type() == PTO_NODE_TYPE::LIST_VARIABLE) {
    //         varList = ((PTO_LIST_VAR*)args[2])->get_var_list();
    //     } else if (args[2]->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
    //         varList = ((PTO_TUPLE_VAR*)args[2])->get_var_list();
    //     } else {
    //         SPDLOG_ERROR("Unexpected Error");
    //         return;
    //     }
    //     for (std::size_t i = 0; i < varList.size(); ++i) {
    //         varList[i]->dump_to_pyTorch(0, fout);
    //         fout << ": ";
    //         varList[i]->dump_to_pyTorch(0, fout);
    //         fout << " + ";
    //         args[1]->dump_to_pyTorch(0, fout);
    //         fout << ".shape[" << i << "]";

    //         if (i != varList.size() - 1)
    //             fout << ", ";
    //     }
        
    //     fout << "] = ";
    //     args[1]->dump_to_pyTorch(0, fout);
    // }
    // else {
    //     fout << indent;
    //     lhs->convert_to_triton_host(0, fout);
    //     fout << " = ";
    //     if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.tensor.create") {
    //         deviceMemoryPtr.insert(lhs->to_string());
    //     }
    //     value->convert_to_triton_host(0, fout);
    // }
}

void PTO_VARIABLE::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // std::string indent(depth, '\t');
    // fout << indent << varName;
}

void PTO_CALL::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // std::string indent(depth, '\t');
    // fout << indent;

    // if (funcName.substr(0, 5) == "self.") {
    //     // 从这调用的说明grid大小是1
    //     fout << funcName.substr(5) << "[(1, )](";
        
    //     // 先确定入参
    //     for (const auto& arg : arguments) {
    //         fout << arg->to_string() << ", ";
    //         if (arg->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
    //             // 必须是二维
    //             if (arg->get_data_type().shape.size() != 2) {
    //                 SPDLOG_ERROR("Only support two-dimensional tensor");
    //                 return;
    //             }
    //             fout << arg->to_string() << ".stride(0), " << arg->to_string() << ".stride(1), ";
    //         }
    //     }

    //     // 输出结果的参数在调用这个函数的地方处理
    // }
    // else if (funcName == "pypto.language.tensor.create") {
    //     fout << "torch.empty(";
    //     arguments[0]->dump_to_pyTorch(0, fout);
    //     fout << ", ";
    //     arguments[1]->dump_to_pyTorch(0, fout);
    //     fout << ", ";
    //     arguments[2]->dump_to_pyTorch(0, fout);
    //     fout << ", ";
    //     fout << "device = 'cuda')";
    // }
    // else {
    //     this->dump_to_pyTorch(0, fout);
    // }

}

void PTO_FOR_LOOP::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // std::string indent(depth, '\t');

    // // 特殊情况处理
    // if (info->get_func_name() == "pypto.language.parallel" && statements.size() == 1) {
    //     // 有可能是kernel调用
    //     if (statements[0]->type() == PTO_NODE_TYPE::ASSIGNMENT && ((PTO_ASSIGNMENT*)statements[0])->get_value()->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)((PTO_ASSIGNMENT*)statements[0])->get_value())->get_func_name().substr(0, 5) == "self.") {
    //         auto lhs = ((PTO_ASSIGNMENT*)statements[0])->get_lhs();
    //         auto kernelPtr = (PTO_CALL*)((PTO_ASSIGNMENT*)statements[0])->get_value();

    //         if (lhs->type() == PTO_NODE_TYPE::VARIABLE || lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
    //             const auto& lhsDataType = lhs->get_data_type();

    //             if (lhsDataType.shape.size() != 2) {
    //                 SPDLOG_ERROR("Only support two-dimensional tensor");
    //                 return;
    //             }

    //             std::string lhsTensorType;
    //             if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::BF16) {
    //                 lhsTensorType = "bfloat16";
    //             }
    //             else if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::FP32) {
    //                 lhsTensorType = "float32";
    //             }
    //             else {
    //                 SPDLOG_ERROR("Unsupported data type");
    //                 return;
    //             }

    //             create_device_memory_on_host(indent, fout, lhs->to_string(), lhsDataType.shape, lhsTensorType);

    //             fout << indent;
    //             fout << kernelPtr->get_func_name().substr(5) << "[(";

    //             // 确认不同axis的具体数值
    //             if (programIdArg.find(kernelPtr->get_func_name().substr(5)) == programIdArg.end()) {
    //                 SPDLOG_ERROR("Unexpected Error");
    //                 return;
    //             }

    //             auto& programIDInfo = programIdArg[kernelPtr->get_func_name().substr(5)];
    //             std::vector<int> grid(programIDInfo.size(), 0);

    //             // 循环变量的维度
    //             grid[programIDInfo[iter->to_string()].axis] = std::stoi(info->get_arguments()[0]->to_string());

    //             for (const auto& id : programIDInfo) {
    //                 if (id.second.range != -1) {
    //                     grid[id.second.axis] = id.second.range;
    //                 }
    //             }

    //             for (const auto& v : grid) {
    //                 fout << v << ", ";
    //             }
    //             fout << ")](";

    //             // 处理入参
    //             for (const auto& arg : kernelPtr->get_arguments()) {
    //                 if (programIDInfo.find(arg->to_string()) != programIDInfo.end()) {
    //                     continue;
    //                 }

    //                 if (arg->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
    //                     if (arg->get_data_type().shape.size() != 2) {
    //                         SPDLOG_ERROR("Only support two-dimensional tensor");
    //                         return;
    //                     }
    //                     fout << arg->to_string() << ", " << arg->to_string() << ".stride(0), " << arg->to_string() << ".stride(1), ";
    //                 }
    //                 else if (arg->get_data_type().kind == PTO_TYPE_KIND::INT32) {
    //                     fout << arg->to_string() << ", ";
    //                 }
    //                 else {
    //                     SPDLOG_ERROR("Unsupported type {}", arg->get_data_type().to_string());
    //                 }
    //             }

    //             if (kernelOutputPtr.find(kernelPtr->get_func_name().substr(5)) != kernelOutputPtr.end() && kernelOutputPtr[kernelPtr->get_func_name().substr(5)].find(0) != kernelOutputPtr[kernelPtr->get_func_name().substr(5)].end()) {
    //                 // 处理输出
    //                 fout << lhs->to_string() << ", " << lhs->to_string() << ".stride(0), " << lhs->to_string() << ".stride(1)";
    //             }
    //             fout << ")";
    //         }
    //         else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
    //             // 要求每个子类型都是tensor
    //             for (const auto& var : ((PTO_TUPLE_VAR*)lhs)->get_var_list()) {
    //                 const auto& lhsDataType = var->get_data_type();

    //                 if (lhsDataType.shape.size() != 2) {
    //                     SPDLOG_ERROR("Only support two-dimensional tensor");
    //                     return;
    //                 }

    //                 std::string lhsTensorType;
    //                 if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::BF16) {
    //                     lhsTensorType = "bfloat16";
    //                 }
    //                 else if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::FP32) {
    //                     lhsTensorType = "float32";
    //                 }
    //                 else {
    //                     SPDLOG_ERROR("Unsupported data type");
    //                     return;
    //                 }

    //                 create_device_memory_on_host(indent, fout, var->to_string(), lhsDataType.shape, lhsTensorType);

    //             }

    //             fout << indent << kernelPtr->get_func_name().substr(5) << "[(";

    //             // 确认不同axis的具体数值
    //             if (programIdArg.find(kernelPtr->get_func_name().substr(5)) == programIdArg.end()) {
    //                 SPDLOG_ERROR("Unexpected Error");
    //                 return;
    //             }

    //             auto& programIDInfo = programIdArg[kernelPtr->get_func_name().substr(5)];
    //             std::vector<int> grid(programIDInfo.size(), 0);

    //             // 循环变量的维度
    //             grid[programIDInfo[iter->to_string()].axis] = std::stoi(info->get_arguments()[0]->to_string());

    //             for (const auto& id : programIDInfo) {
    //                 if (id.second.range != -1) {
    //                     grid[id.second.axis] = id.second.range;
    //                 }
    //             }

    //             for (const auto& v : grid) {
    //                 fout << v << ", ";
    //             }
    //             fout << ")](";

    //             // 处理入参
    //             for (const auto& arg : kernelPtr->get_arguments()) {
    //                 if (programIDInfo.find(arg->to_string()) != programIDInfo.end()) {
    //                     continue;
    //                 }

    //                 if (arg->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
    //                     if (arg->get_data_type().shape.size() != 2) {
    //                         SPDLOG_ERROR("Only support two-dimensional tensor");
    //                         return;
    //                     }
    //                     fout << arg->to_string() << ", " << arg->to_string() << ".stride(0), " << arg->to_string() << ".stride(1), ";
    //                 }
    //                 else if (arg->get_data_type().kind == PTO_TYPE_KIND::INT32) {
    //                     fout << arg->to_string() << ", ";
    //                 }
    //                 else {
    //                     SPDLOG_ERROR("Unsupported type {}", arg->get_data_type().to_string());
    //                 }
    //             }

    //             if (kernelOutputPtr.find(kernelPtr->get_func_name().substr(5)) != kernelOutputPtr.end() && kernelOutputPtr[kernelPtr->get_func_name().substr(5)].find(0) != kernelOutputPtr[kernelPtr->get_func_name().substr(5)].end()) {
    //                 // 处理输出
    //                 fout << lhs->to_string() << ", " << lhs->to_string() << ".stride(0), " << lhs->to_string() << ".stride(1)";
    //             }
    //             fout << ")";
    //         }
    //         else {
    //             SPDLOG_ERROR("Unimplemented");
    //             return;
    //         }
            

    //         return;
    //     }
    // }

    // // 处理成串行的for loop
    // fout << indent << "for ";
    // iter->dump_to_pyTorch(0, fout);
    // fout << " in range(";
    // for (const auto& arg : info->get_arguments()) {
    //     if (arg->type() != PTO_NODE_TYPE::KEYWORD) {
    //         arg->dump_to_pyTorch(0, fout);
    //         fout << ", ";
    //     }
    // }
    // fout << "):" << std::endl;
    // for (const auto& s : statements) {
    //     s->convert_to_triton_host(depth + 1, fout);
    //     fout << std::endl;
    // }
}

void PTO_BINARY_OP::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // // 直接调用torch的版本
    // this->dump_to_pyTorch(depth, fout);
}

void PTO_RETURN::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // // 直接调用torch的版本
    // this->dump_to_pyTorch(depth, fout);
}
    
} // namespace pto_parser
