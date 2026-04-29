#include "ptoNode.hh"
#include "logger.hh"
#include <sstream>

namespace pto_parser
{
// 用于记录program ID信息
struct PROGRAM_ID{
    int axis; // grid的第几个维度？
    int range; // kernel函数内部的循环范围，对所有kernel函数的调用是相同的值；对于host内产生的programID，则为-1
    std::unordered_map<const PTO_CALL*, int> rangePerCall; // Host内每次调用kernel函数可能有不同的循环次数，在此处记录

    PROGRAM_ID()
        : axis(-1),
          range(-1),
          rangePerCall()
    {}
};

// 记录每个函数名，有哪些变量是循环变量
// 第一个index是kernel函数的名字，第二个index是入参的序号
static std::unordered_map<std::string, std::unordered_map<int, struct PROGRAM_ID>> programIDInfo;

////////////////////////////////////////////////
// 以下是用于处理device memory加载的辅助数据结构
////////////////////////////////////////////////
// 入参里的tensor都是deviceMemory的指针
static std::unordered_map<std::string, std::vector<std::string>> deviceMemoryPtr;
// 记录当前kernel函数内哪些devicememory指针被直接使用
static std::unordered_set<std::string> directUsedDeviceMemory;
// 记录每个kernel函数的第几个output指针需要在函数的参数列表里新建
static std::unordered_map<std::string, std::unordered_set<int>> kernelOutputPtr;
// 记录当前kernel函数的返回变量名
static std::vector<std::string> curOutputName;
// 转换成triton的过程中，不是一对一转化，可能需要生成新的中间变量，所以这里记录kernenl函数内部用到的变量名，避免错误
static std::unordered_set<std::string> kernelUsedVarName;
// 记录存在于local内存的变量，作为是否需要添加tl.store的依据
static std::unordered_set<std::string> localVar;


////////////////////////////////////////////
// 以下是用于处理tensor的尺寸不是二的冥的情况
////////////////////////////////////////////
// 记录每个tensor在PTO源码中的shape，可以不是二的冥，是有效数据的大小
static std::unordered_map<std::string, std::vector<int>> tensorOriginalShape;
// 记录每个tensor在生成的triton代码中的shape，必须是二的冥，包含无效数据，无效数据默认都设为0
static std::unordered_map<std::string, std::vector<int>> tensorActualShape;
// 如果tensor的某个维度是动态维度，则在original shape中用-1表示，使用的变量在这个里面记录
static std::unordered_map<std::string, std::vector<std::string>> tensorDynamicShape;

static inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static inline int next_power_of_two(int n) {
    n--; 
    
    // 把最高位的 1 向右“抹平”
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    
    // 此时 n 的二进制变成了 000...01111111
    // 加 1 就会进位，变成 000...10000000，完美得到 2 的幂
    return n + 1;
}

static const std::string generate_offset(PTO_LIST_VAR* subShape, PTO_LIST_VAR* startPos, const std::string& tensorName) {
    std::stringstream ret;
    ret << "(" << startPos->get_var_list()[0]->to_string() << " + tl.arange(0, " << subShape->get_var_list()[0]->to_string() << "))[:, None] * " << deviceMemoryPtr[tensorName][1] <<
        " + (" << startPos->get_var_list()[1]->to_string() << " + tl.arange(0, " << subShape->get_var_list()[1]->to_string() << "))[None, :] * " << deviceMemoryPtr[tensorName][2];

    return ret.str();
}

static const std::string generate_offset(const std::vector<int>& subShape, PTO_LIST_VAR* startPos, const std::string& tensorName) {
    std::stringstream ret;
    ret << "(" << startPos->get_var_list()[0]->to_string() << " + tl.arange(0, " << subShape[0] << "))[:, None] * " << deviceMemoryPtr[tensorName][1] <<
        " + (" << startPos->get_var_list()[1]->to_string() << " + tl.arange(0, " << subShape[1] << "))[None, :] * " << deviceMemoryPtr[tensorName][2];

    return ret.str();
}

static const std::string generate_offset(const std::vector<int>& subShape, const std::string& tensorName) {
    std::stringstream ret;
    ret << "(tl.arange(0, " << subShape[0] << "))[:, None] * " << deviceMemoryPtr[tensorName][1] <<
        " + (tl.arange(0, " << subShape[1] << "))[None, :] * " << deviceMemoryPtr[tensorName][2];

    return ret.str();
}

static const std::string generate_mask(const std::vector<int>& subShape, PTO_LIST_VAR* validRegion) {
    if (subShape.size() != 2 || validRegion->get_var_list().size() != 2) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return "";
    }

    std::stringstream ret;

    ret << "((tl.arange(0, " << subShape[0] << "))[:, None] < " << validRegion->get_var_list()[0]->to_string() << ") & " <<
           "((tl.arange(0, " << subShape[1] << "))[None, :] < " << validRegion->get_var_list()[1]->to_string() << ")";

    return ret.str();
}

static const std::string generate_mask(PTO_LIST_VAR* subShape, PTO_LIST_VAR* startPos, const std::string& srcTensor) {
    if (subShape->get_var_list().size() != 2 || startPos->get_var_list().size() != 2) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return "";
    }

    std::stringstream ret;
    ret << "((tl.arange(0, " << srcTensor << ".shape(0)) >= " << startPos->get_var_list()[0]->to_string() << ") & (tl.arange(0, " << 
                                srcTensor << ".shape(0)) < " << startPos->get_var_list()[0]->to_string() << " + " << subShape->get_var_list()[0]->to_string() << "))[:, None] & " <<
           "((tl.arange(0, " << srcTensor << ".shape(1)) >= " << startPos->get_var_list()[1]->to_string() << ") & (tl.arange(0, " << 
                                srcTensor << ".shape(1)) < " << startPos->get_var_list()[1]->to_string() << " + " << subShape->get_var_list()[1]->to_string() << "))[None, :]";

    return ret.str();
}

static void gen_triton_load(std::ofstream& fout, const int& depth, const std::string& varName) {
    const std::string indent(depth, '\t');

    if (deviceMemoryPtr.find(varName) == deviceMemoryPtr.end()) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    if (tensorOriginalShape.find(varName) == tensorOriginalShape.end()) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    if (tensorDynamicShape.find(varName) == tensorDynamicShape.end()) {
        SPDLOG_ERROR("Unexpected Error");
        return;
    }

    // 暂时只支持二维tensor
    const auto& originalShape = tensorOriginalShape[varName];
    if (originalShape.size() != 2) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return;
    }

    // shape都是二的冥
    if (is_power_of_two(originalShape[0]) && is_power_of_two(originalShape[1])) {
        // actual shape和original shape相同
        tensorActualShape[varName] = originalShape;

        // 只用生成offset，不用生成mask
        auto offsetName = varName + "_offset";
        while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
            offsetName += '_';
        }
        kernelUsedVarName.insert(offsetName);
        fout << indent << "# 实际尺寸和原始尺寸一致, 不需要mask" << std::endl;
        fout << indent << offsetName << " = " << generate_offset(originalShape, varName) << std::endl;
        fout << indent << varName << " = tl.load(" << deviceMemoryPtr[varName][0] << " + " << offsetName << ")" << std::endl;
    } else {
        SPDLOG_ERROR("Unimplemented");
        return;
    }
}

static void gen_triton_load(std::ofstream& fout, const int& depth, const std::string& dstName, const std::string& srcName, PTO_LIST_VAR* subShape, PTO_LIST_VAR* startPos, PTO_LIST_VAR* tensorView) {
    const std::string indent(depth, '\t');

    // 强制要求是二维tensor
    if (subShape->get_var_list().size() != 2 || subShape->get_var_list()[0]->type() != PTO_NODE_TYPE::INT_CONSTANT || subShape->get_var_list()[1]->type() != PTO_NODE_TYPE::INT_CONSTANT) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return;
    }

    if (startPos->get_var_list().size() != 2) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return;
    }

    const auto& dstShape = subShape->get_var_list();

    if (tensorView != nullptr) {
        // 有效数据的范围是基于tensorView确定，subShape的尺寸应当大于tensorView，这里无法做检查，因为tensorView的值可能是动态确定的
        // 基于subShape生成对应的actual size，基于tensorView生成original size
        
        // 先基于subShape生成offset，这里需要判断是否是2的冥
        std::string offsetName = dstName + "_offset";
        while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
            offsetName += '_';
        }
        kernelUsedVarName.insert(offsetName);
        if (is_power_of_two(std::stoi(dstShape[0]->to_string())) && is_power_of_two(std::stoi(dstShape[1]->to_string()))) {
            fout << indent << offsetName << " = " << generate_offset(subShape, startPos, srcName) << std::endl;

            tensorActualShape[dstName] = {std::stoi(dstShape[0]->to_string()), std::stoi(dstShape[1]->to_string())};
        }
        else {
            // 不应该有动态shape，dst的shape应当是静态确定的
            if (std::stoi(dstShape[0]->to_string()) == -1 || std::stoi(dstShape[1]->to_string()) == -1) {
                SPDLOG_ERROR("Unexpected Error");
                return;
            }

            std::vector<int> actualShape = {next_power_of_two(std::stoi(dstShape[0]->to_string())), next_power_of_two(std::stoi(dstShape[1]->to_string()))};

            fout << indent << "# " << dstName << "的shape从[" << dstShape[0]->to_string() << ", " << dstShape[1]->to_string() << "]拓展成了[" << actualShape[0] << ", " << actualShape[1] << "], 多出来的数据填0" << std::endl;
            fout << indent << offsetName << " = " << generate_offset(actualShape, startPos, srcName) << std::endl;

            tensorActualShape[dstName] = actualShape;
        }

        // 根据tensorView包含的有效数据范围生成Mask
        std::string maskName = dstName + "_mask";
        while (kernelUsedVarName.find(maskName) != kernelUsedVarName.end()) {
            maskName += "_";
        }
        kernelUsedVarName.insert(maskName);

        fout << indent << "# 基于有效数据范围[" << tensorView->get_var_list()[0]->to_string() << ", " << tensorView->get_var_list()[1]->to_string() << "]生成Mask" << std::endl;
        fout << indent << maskName << " = " << generate_mask(tensorActualShape[dstName], tensorView) << std::endl;

        // 记录这个tensor有数据的范围
        std::vector<int> validShape;
        std::vector<std::string> dynamicShape;
        if (tensorView->get_var_list()[0]->type() == PTO_NODE_TYPE::INT_CONSTANT) {
            validShape.emplace_back(std::stoi(tensorView->get_var_list()[0]->to_string()));
            dynamicShape.emplace_back("");
        } else {
            // 是一个动态shape
            validShape.emplace_back(-1);
            dynamicShape.emplace_back(tensorView->get_var_list()[0]->to_string());
        }
        if (tensorView->get_var_list()[1]->type() == PTO_NODE_TYPE::INT_CONSTANT) {
            validShape.emplace_back(std::stoi(tensorView->get_var_list()[1]->to_string()));
            dynamicShape.emplace_back("");
        } else {
            // 是一个动态shape
            validShape.emplace_back(-1);
            dynamicShape.emplace_back(tensorView->get_var_list()[1]->to_string());
        }

        // 生成load
        fout << indent << dstName << " = tl.load(" << deviceMemoryPtr[srcName][0] << " + " << offsetName << ", mask=" << maskName << ", other=0.0)";
    } else {
        // 没有tensorView，所以有效数据的范围就是根据subShape生成
        if (is_power_of_two(std::stoi(dstShape[0]->to_string())) && is_power_of_two(std::stoi(dstShape[1]->to_string()))) {
            // subShape是2的冥，所有有效数据和实际tensor大小是一致的
            fout << indent << "# 因为输入的PTO源码没有边界检查, 所以没有为这个tl.load生成Mask" << std::endl;
            // 生成offset
            std::string offsetName = dstName + "_offset";
            while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
                offsetName += '_';
            }
            kernelUsedVarName.insert(offsetName);

            fout << indent << offsetName << " = " << generate_offset(subShape, startPos, srcName) << std::endl;

            // 不生成mask
            fout << indent << dstName << " = tl.load(" << deviceMemoryPtr[srcName][0] << " + " << offsetName << ")";

            // 存下这个tensor的尺寸
            tensorOriginalShape[dstName] = {std::stoi(dstShape[0]->to_string()), std::stoi(dstShape[1]->to_string())};
            tensorActualShape[dstName] = {std::stoi(dstShape[0]->to_string()), std::stoi(dstShape[1]->to_string())};
            tensorDynamicShape[dstName] = {"", ""};
        } else {
            SPDLOG_ERROR("Unimplemented for not power of two");
        }
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

    // 生成host代码
    ptr->convert_to_triton_host(1, fout);
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
    localVar.clear();

    tensorOriginalShape.clear();
    tensorActualShape.clear();
    tensorDynamicShape.clear();

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

            // 记录这个变量的实际大小
            tensorOriginalShape[arguments[i]->to_string()] = argType.shape;
            tensorDynamicShape[arguments[i]->to_string()] = argType.dynamicShape;
        }
    }

    // 扫描所有statement，完成三项任务
    // 1. 收集使用过的变量名
    // 2. 收集被直接使用的deviceMemory的名字
    // 3. 收集返回值的名字，判断是否需要增加入参
    for (const auto& s : statements) {
        s->collect_triton_kernel_info();
    }

    // 这里记录下入参的类型，后面生成tl.load时需要
    std::unordered_map<std::string, PTO_TYPE> deviceMemoryType;

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
            fout << indent << "\t" << convertName[0] << ", " << convertName[1] << ", " << convertName[2] << ", " << std::endl;

            deviceMemoryType[arguments[i]->to_string()] = argType;
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

    // 对于直接只用的device memory生成tl.load
    for (const auto& var : directUsedDeviceMemory) {
        gen_triton_load(fout, depth + 1, var);
    }
    
    // 逐个生成statement，在这个过程中需要记录有哪些variable是在local memory中，如果这个variable出现在return中时，
    // 则需要生成tl.store命令，注意，assemble指令会转换成tl.store,这时这个variable则认为被存到device memory中了
    for (const auto& it : statements) {
        it->convert_to_triton_kernel(depth + 1, fout);
        if (it->type() != PTO_NODE_TYPE::FOR_LOOP && it->type() != PTO_NODE_TYPE::IF) {
            fout << std::endl;
        }
    }
    fout << std::endl;
}

void PTO_ASSIGNMENT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');

    // 如果右侧是funcCall,则有几个特殊情况需要处理
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL) {
        auto funcPtr = (PTO_CALL*)value;
        if (funcPtr->get_func_name() == "pypto.language.tensor.slice") {
            const auto& tensorName = funcPtr->get_arguments()[0]->to_string();

            // 强制要求是二维tensor
            const auto subShape = (PTO_LIST_VAR*)funcPtr->get_arguments()[1];
            const auto startPos = (PTO_LIST_VAR*)funcPtr->get_arguments()[2];

            PTO_LIST_VAR* tensorView = nullptr;
            if (funcPtr->get_arguments().size() == 4) {
                tensorView = (PTO_LIST_VAR*)funcPtr->get_arguments()[3];
            }

            if (subShape->get_var_list().size() != 2) {
                SPDLOG_ERROR("Only support two-dimensional tensor");
                return;
            }

            if (deviceMemoryPtr.find(tensorName) != deviceMemoryPtr.end()) {
                // 是从device memory加载回来的local变量
                // 在这个statement时，虽然这个变量存在于local memory中，但其值没有发生改变，所以先不需要存于localVar中
                gen_triton_load(fout, depth, lhs->to_string(), tensorName, subShape, startPos, tensorView);
            }
            else {
                if (tensorView != nullptr) {
                    // 在qwen3 decode layer中没遇到该情况
                    SPDLOG_ERROR("Unexpected Error");
                    return;
                }

                // 是从local memory到local memory的变量赋值
                // 认为左侧的变量是新的寄存器的值
                localVar.insert(lhs->to_string());

                // 如果subShape不是2的冥的话，使用tl.where处理
                if (subShape->get_var_list()[0]->type() != PTO_NODE_TYPE::INT_CONSTANT || !is_power_of_two(std::stoi(subShape->get_var_list()[0]->to_string())) || 
                    subShape->get_var_list()[1]->type() != PTO_NODE_TYPE::INT_CONSTANT || !is_power_of_two(std::stoi(subShape->get_var_list()[1]->to_string()))) {
                    
                    // 需要先生成mask
                    std::string maskName = lhs->to_string() + "_mask";
                    while (kernelUsedVarName.find(maskName) != kernelUsedVarName.end()) {
                        maskName += '_';
                    }
                    kernelUsedVarName.insert(maskName);
                    
                    fout << indent << "# 基于初始位置[" << startPos->get_var_list()[0]->to_string() << ", " << startPos->get_var_list()[1]->to_string() << "]和tensor形状[" <<
                                                        subShape->get_var_list()[0]->to_string() << ", " << subShape->get_var_list()[1]->to_string() << "]生成tl.where使用的Mask" << std::endl;
                    fout << indent << maskName << " = " << generate_mask(subShape, startPos, tensorName) << std::endl;
                    fout << indent << "# 使用where替换slice, Mask以外的数据填0" << std::endl;
                    fout << indent << lhs->to_string() << " = tl.where(" << maskName << ", " << tensorName << ", 0.0)";
                }
                else {
                    // 直接切分
                    SPDLOG_ERROR("Slice from non-device memory ptr at line {} cannot be converted to triton", row_);
                    return;
                }
            }
            // 处理完毕
            return;
        }
        else if (funcPtr->get_func_name() == "pypto.language.tensor.assemble") {
            if (deviceMemoryPtr.find(lhs->to_string()) != deviceMemoryPtr.end()) {
                // 对于A = pl.assemble(B, C, [startPos])
                // 如果B是个存在于local memory中的变量，则无论A是否等于B，等先需要一个tl.store将B存入A的位置
                // 如果B是个存在于device memory中的变量，但A不等于B，也需要一个tl.store将B存入A的位置
                // 之后再计算C的offset，把C存入A的对应位置
                // 这两部操作后，我们认为A的值已经存在于device memory中了，所以可以从localVar中删除A
                // 注意，我们不能删除B
                const auto src0Name = funcPtr->get_arguments()[0]->to_string();
                if (localVar.find(src0Name) != localVar.end() || src0Name != lhs->to_string()) {
                    // 需要一个offset
                    const auto& src0Type = funcPtr->get_arguments()[0]->get_data_type();
                    if (src0Type.kind != PTO_TYPE_KIND::TENSOR || src0Type.shape.size() != 2) {
                        SPDLOG_ERROR("Only support two-dimensional tensor");
                        return;
                    }
                    auto offsetName = src0Name + "_offset";
                    while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
                        offsetName += "_";
                    }
                    kernelUsedVarName.insert(offsetName);

                    fout << indent << offsetName << " = " << generate_offset(src0Type.shape, src0Name) << std::endl;
                    fout << indent << "tl.store(" << deviceMemoryPtr[lhs->to_string()][0] << " + " << offsetName << ", " << src0Name << ")" << std::endl;
                }

                // 拿到subShape
                const auto& argDataType = funcPtr->get_arguments()[1]->get_data_type();
                if (argDataType.kind != PTO_TYPE_KIND::TENSOR || argDataType.shape.size() != 2) {
                    SPDLOG_ERROR("Only support two-dimensional tensor");
                    return;
                }
                auto offsetName = lhs->to_string() + "_offset";
                while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
                    offsetName += '_';
                }
                kernelUsedVarName.insert(offsetName);

                fout << indent << offsetName << " = " << generate_offset(argDataType.shape, (PTO_LIST_VAR*)funcPtr->get_arguments()[2], lhs->to_string()) << std::endl;
                fout << indent << "tl.store(" << deviceMemoryPtr[lhs->to_string()][0] << " + " << offsetName << ", " << funcPtr->get_arguments()[1]->to_string() << ")";
            }
            else {
                // 需要知道第一个参数和第二个参数的实际尺寸
                // 做一个强制要求，要求某个维度必须是1
                const auto& lhsDataType = lhs->get_data_type();
                if (lhsDataType.kind != PTO_TYPE_KIND::TENSOR || lhsDataType.shape.size() != 2) {
                    SPDLOG_ERROR("Only support two-dimension tensor");
                    return;
                }
                if (lhsDataType.shape[0] != 1 && lhsDataType.shape[1] != 1) {
                    SPDLOG_ERROR("One dimension must be 1");
                    return;
                }
                
                // 先做一个赋值
                fout << indent << lhs->to_string() << " = " << funcPtr->get_arguments()[0]->to_string() << std::endl;

                // 拿到第二个参数的维度
                const auto& valueDataType = funcPtr->get_arguments()[1]->get_data_type();

                if (valueDataType.kind != PTO_TYPE_KIND::TENSOR) {
                    SPDLOG_ERROR("Unexpected Error");
                    return;
                }

                // 先沿着1的维度将lhs和第二个参数压为1维tensor
                // 使用tl.view将两个变量都压到1维
                if (lhsDataType.shape[0] == 1) {
                    fout << indent << lhs->to_string() << " = tl.view(" << lhs->to_string() << ", (" << lhsDataType.shape[1] << ",))" << std::endl;
                    if (valueDataType.shape[0] != 1) {
                        SPDLOG_ERROR("Unexpected Error");
                        return;
                    }
                    fout << indent << funcPtr->get_arguments()[1]->to_string() << " = tl.view(" << funcPtr->get_arguments()[1]->to_string() << ", (" << valueDataType.shape[1] << ",))" << std::endl;
                } else {
                    fout << indent << lhs->to_string() << " = tl.view(" << lhs->to_string() << ", (" << lhsDataType.shape[0] << ",))" << std::endl;
                    if (valueDataType.shape[1] != 1) {
                        SPDLOG_ERROR("Unexpected Error");
                        return;
                    }
                    fout << indent << funcPtr->get_arguments()[1]->to_string() << " = tl.view(" << funcPtr->get_arguments()[1]->to_string() << ", (" << valueDataType.shape[0] << ",))" << std::endl;
                }

                // 使用tl.cat完成一维向量的拼接
                // 强制要求拼接的位置要么在开头，要么在末尾，即可以写成两个向量的拼接
                const auto& startPos = (PTO_LIST_VAR*)funcPtr->get_arguments()[2];
                if (startPos->get_var_list().size() != 2 || startPos->get_var_list()[0]->type() != PTO_NODE_TYPE::INT_CONSTANT || startPos->get_var_list()[1]->type() != PTO_NODE_TYPE::INT_CONSTANT) {
                    SPDLOG_ERROR("Unexpected error");
                    return;
                }

                if (lhsDataType.shape[0] == 1) {
                    if (std::stoi(startPos->get_var_list()[0]->to_string()) != 0) {
                        SPDLOG_ERROR("Unexpected Error");
                        return;
                    }
                    int startPosInt = std::stoi(startPos->get_var_list()[1]->to_string());
                    
                    if (startPosInt == 0) {
                        fout << indent << lhs->to_string() << " = tl.cat(" << funcPtr->get_arguments()[1]->to_string() << ", " << lhs->to_string() << "[" << valueDataType.shape[1] << ": " << lhsDataType.shape[1] << "])" << std::endl;
                    }
                    else if (startPosInt + valueDataType.shape[1] == lhsDataType.shape[1]) {
                        fout << indent << lhs->to_string() << " = tl.cat(" << lhs->to_string() << "[0: " << startPosInt << "], " << funcPtr->get_arguments()[1]->to_string() << ")" << std::endl;
                    }
                    else {
                        SPDLOG_ERROR("Unexpected position for assemble {} {} {}", startPosInt, valueDataType.shape[1], lhsDataType.shape[1]);
                        return;
                    }
                } else {
                    if (std::stoi(startPos->get_var_list()[1]->to_string()) != 0) {
                        SPDLOG_ERROR("Unexpected Error");
                        return;
                    }
                    int startPosInt = std::stoi(startPos->get_var_list()[0]->to_string());

                    if (startPosInt == 0) {
                        fout << indent << lhs->to_string() << " = tl.cat(" << funcPtr->get_arguments()[1]->to_string() << ", " << lhs->to_string() << "[" << valueDataType.shape[0] << ": " << lhsDataType.shape[0] << "])" << std::endl;
                    }
                    else if (startPosInt + valueDataType.shape[0] == lhsDataType.shape[0]) {
                        fout << indent << lhs->to_string() << " = tl.cat(" << lhs->to_string() << "[0: " << startPosInt << "], " << funcPtr->get_arguments()[1]->to_string() << ")" << std::endl;
                    }
                    else {
                        SPDLOG_ERROR("Unexpected position for assemble {} {} {}", startPosInt, valueDataType.shape[0], lhsDataType.shape[0]);
                        return;
                    }
                }

                // 拉回二维
                if (lhsDataType.shape[0] == 1) {
                    fout << indent << lhs->to_string() << " = " << lhs->to_string() << "[None, :]";
                } else {
                    fout << indent << lhs->to_string() << " = " << lhs->to_string() << "[:, None]";
                }

                // 左侧的变量在local memory中的值已经修改
                localVar.insert(lhs->to_string());
            }
            // 处理完毕
            return;
        }
    }

    // 常规赋值语句处理，认为左侧的变量都在local memory中，且值已修改
    localVar.insert(lhs->to_string());
    fout << indent << lhs->to_string() << " = ";
    value->convert_to_triton_kernel(0, fout);
}

void PTO_CALL::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    fout << indent;
    if (funcName == "pypto.language.tensor.create") {
        fout << "tl.zeros(";
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            arguments[i]->convert_to_triton_kernel(0, fout);
            if (i != arguments.size() - 1) {
                fout << ", ";
            }
        }
        fout << ")";
    }
    else if (funcName == "pypto.language.tensor.muls" || funcName == "pypto.language.tensor.mul" || funcName == "pypto.language.tensor.row_expand_mul" || funcName == "pypto.language.tensor.col_expand_mul") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << " * ";
        arguments[1]->convert_to_triton_kernel(0, fout);
    }
    else if (funcName == "pypto.language.tensor.cast" || funcName == "pypto.language.cast") {
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ".to(";
        if (arguments[1]->type() == PTO_NODE_TYPE::KEYWORD) {
            arguments[1]->convert_to_triton_kernel(0, fout);
        }
        else if (arguments[1]->to_string() == "pypto.language.INDEX") {
            fout << "tl.int32";
        }
        else {
            SPDLOG_ERROR("Unimplemented arg {}", arguments[1]->to_string());
        }
        fout << ")";
    }
    else if (funcName == "pypto.language.tensor.row_sum") {
        fout << "tl.sum(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ", axis = 1)[:, None]";
    }
    else if (funcName == "pypto.language.tensor.adds" || funcName == "pypto.language.tensor.add") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << " + ";
        arguments[1]->convert_to_triton_kernel(0, fout);
    }
    else if (funcName == "pypto.language.tensor.sub" || funcName == "pypto.language.tensor.row_expand_sub") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << " - ";
        arguments[1]->convert_to_triton_kernel(0, fout);
    }
    else if (funcName == "pypto.language.tensor.row_expand_div") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << " / ";
        arguments[1]->convert_to_triton_kernel(0, fout);
    }
    else if (funcName == "pypto.language.tensor.rsqrt") {
        fout << "tl.math.rsqrt(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ")";
    }
    else if (funcName == "pypto.language.tensor.exp") {
        fout << "tl.exp(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ")";
    }
    else if (funcName == "pypto.language.tensor.matmul") {
        fout << "tl.dot(";
        // a_trans是否为真
        bool a_trans = false, b_trans = false;
        for (const auto& arg : arguments) {
            if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "a_trans") {
                if (((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
                    a_trans = true;
                }
            }
            if (arg->type() == PTO_NODE_TYPE::KEYWORD && ((PTO_KEYWORD*)arg)->get_keyword() == "b_trans") {
                if (((PTO_KEYWORD*)arg)->get_value()->to_string() == "true") {
                    b_trans = true;
                }
            }
        }

        if (a_trans) {
            fout << "tl.trans(";
            arguments[0]->convert_to_triton_kernel(0, fout);
            fout << "), ";
        } else {
            arguments[0]->convert_to_triton_kernel(0, fout);
            fout << ", ";
        }
        if (b_trans) {
            fout << "tl.trans(";
            arguments[1]->convert_to_triton_kernel(0, fout);
            fout <<")";
        } else {
            arguments[1]->convert_to_triton_kernel(0, fout);
        }
        fout << ")";
    }
    else if (funcName == "pypto.language.tensor.row_max") {
        fout << "tl.max(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ", axis = 1)[:, None]";
    }
    else if (funcName == "pypto.language.tensor.maximum") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        fout << "tl.maximum(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ", ";
        arguments[1]->convert_to_triton_kernel(0, fout);
        fout << ")";
    }
    else if (funcName == "pypto.language.min") {
        if (arguments.size() != 2) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        fout << "tl.minimum(";
        arguments[0]->convert_to_triton_kernel(0, fout);
        fout << ", ";
        arguments[1]->convert_to_triton_kernel(0, fout);
        fout << ")";
    }
    else {
        SPDLOG_ERROR("Unimplemented function name {}", funcName);
    }

}

void PTO_TUPLE_VAR::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    fout << indent << "(";
    for (std::size_t i = 0; i < varList.size(); ++i) {
        varList[i]->convert_to_triton_kernel(0, fout);
        if (i != varList.size() - 1) {
            fout << ", ";
        }
    }
    fout << ")";
}

void PTO_LIST_VAR::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    fout << indent << "[";
    for (std::size_t i = 0; i < varList.size(); ++i) {
        varList[i]->convert_to_triton_kernel(0, fout);
        if (i != varList.size() - 1) {
            fout << ", ";
        }
    }
    fout << "]";
}

void PTO_KEYWORD::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    if (keyword == "dtype") {
        fout << indent << keyword << " = ";

        std::string temp = value->to_string();

        if (temp == "pypto.language.FP32") {
            fout << "tl.float32";
        }
        else if (temp == "pypto.language.BF16") {
            fout << "tl.bfloat16";
        }
        else {
            SPDLOG_ERROR("Unimplemented data type {}", temp);
        }
    }
    else if (keyword == "layout") {
        // 忽略
    }
    else if (keyword == "target_type") {
        std::string temp = value->to_string();
        if (temp == "pypto.language.FP32") {
            fout << "tl.float32";
        }
        else if (temp == "pypto.language.BF16") {
            fout << "tl.bfloat16";
        }
        else {
            SPDLOG_ERROR("Unimplemented data type {}", temp);
        }
    }
    else {
        SPDLOG_ERROR("Unimplemented keyword {}" , keyword);
    }
}

void PTO_INT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    fout << std::string(depth, '\t') << value;
}

void PTO_FLOAT::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    fout << std::string(depth, '\t') << value;
}

void PTO_BOOL::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    if (value) {
        fout << std::string(depth, '\t') << "True";
    }
    else {
        fout << std::string(depth, '\t') << "False";
    }
}

void PTO_VARIABLE::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    fout << std::string(depth, '\t') << varName;
}

void PTO_FOR_LOOP::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    // 处理成串行的range
    std::string indent(depth, '\t');
    fout << indent << "for ";
    iter->convert_to_triton_kernel(0, fout);
    fout << " in range(";
    info->get_arguments()[0]->convert_to_triton_kernel(0, fout);
    fout << "):" << std::endl;
    for (const auto& s : statements) {
        s->convert_to_triton_kernel(depth + 1, fout);
        fout << std::endl;
    }
}

void PTO_BINARY_OP::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    fout << indent << "(";
    lhs->convert_to_triton_kernel(0, fout);

    switch (op) {
        case PTO_OPERATOR::ADD: fout << " + "; break;
        case PTO_OPERATOR::SUB: fout << " - "; break;
        case PTO_OPERATOR::FLOOR_DIV: fout << " // "; break;
        case PTO_OPERATOR::MUL: fout << " * "; break;
        case PTO_OPERATOR::EQUAL: fout << " == "; break;
    }
    rhs->convert_to_triton_kernel(0, fout);
    fout << ")";
}

void PTO_RETURN::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    for (const auto& val : returnVal) {
        if (deviceMemoryPtr.find(val->to_string()) == deviceMemoryPtr.end()) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }

        // 需要做一个判断，返回的值是否只存在于local memory中，可能之前已经有assemble语句将值存回device里
        if (localVar.find(val->to_string()) == localVar.end()) {
            continue;
        }
        // 需要生成对应的offset
        std::string offsetName = val->to_string() + "_offset";
        while (kernelUsedVarName.find(offsetName) != kernelUsedVarName.end()) {
            offsetName += '_';
        }
        kernelUsedVarName.insert(offsetName);

        // 必须是二维tensor
        const auto& valDataType = val->get_data_type();
        if (valDataType.kind != PTO_TYPE_KIND::TENSOR || valDataType.shape.size() != 2) {
            SPDLOG_ERROR("Only support two-dimension tensor");
            return;
        }

        fout << indent << offsetName << " = " << generate_offset(valDataType.shape, val->to_string()) << std::endl;

        fout << indent << "tl.store(" << deviceMemoryPtr[val->to_string()][0] << " + " << offsetName << ", " << val->to_string() << ")";
    }

}

void PTO_IF::convert_to_triton_kernel(int depth, std::ofstream& fout) {
    std::string indent(depth, '\t');
    fout << indent << "if ";
    comparator->convert_to_triton_kernel(0, fout);
    fout << ":" << std::endl;
    for (const auto& s : ifStatement) {
        s->convert_to_triton_kernel(depth + 1, fout);
        fout << std::endl;
    }
    fout << indent << "else:" << std::endl;
    for (const auto& s : elseStatement) {
        s->convert_to_triton_kernel(depth + 1, fout);
        fout << std::endl;
    }
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
    if (deviceMemoryPtr.find(name) != deviceMemoryPtr.end()) return;
    deviceMemoryPtr[name] = std::vector<std::string>();

    if (shape.size() != 2) {
        SPDLOG_ERROR("Only support two-dimensional tensor");
        return;
    }

    fout << indent << name << " = torch.empty([" << shape[0] << ", " << shape[1] << "], " <<
                              "dtype = torch." << type << ", device = 'cuda')" << std::endl;
}

void PTO_FUNC::convert_to_triton_host(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');

    // 记录哪些变量的device memory已经创建，后续不会在重复创建
    deviceMemoryPtr.clear();

    fout << indent << "def " << funcName << "(";
    for (const auto& arg : arguments) {
        fout << arg->to_string() << ", ";

        if (arg->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
            deviceMemoryPtr[arg->to_string()] = std::vector<std::string>();
        }
    }
    fout << "):" << std::endl;


    for (const auto& s : statements) {
        s->convert_to_triton_host(depth + 1, fout);
        if (s->type() != PTO_NODE_TYPE::FOR_LOOP && s->type() != PTO_NODE_TYPE::IF) {
            fout << std::endl;
        }
    }
}

void PTO_ASSIGNMENT::convert_to_triton_host(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');

    // 对于kernel函数的调用需要特殊处理
    if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name().substr(0, 5) == "self.") {
        std::string kernelName = ((PTO_CALL*)value)->get_func_name().substr(5);
        // 先要创建output memory
        if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            const auto& lhsDataType = lhs->get_data_type();
            if (lhsDataType.kind != PTO_TYPE_KIND::TENSOR || lhsDataType.shape.size() != 2) {
                SPDLOG_ERROR("Only support two-dimensional tensor");
                return;
            }
            std::string tensorType = "";
            if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::BF16) {
                tensorType = "bfloat16";
            }
            else if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::FP32) {
                tensorType = "float32";
            }
            else {
                SPDLOG_ERROR("Unsupported data type");
                return;
            }
            create_device_memory_on_host(indent, fout, lhs->to_string(), lhsDataType.shape, tensorType);
        }
        else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            const auto& varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            for (const auto& var : varList) {
                const auto& lhsDataType = var->get_data_type();
                if (lhsDataType.kind != PTO_TYPE_KIND::TENSOR || lhsDataType.shape.size() != 2) {
                    SPDLOG_ERROR("Only support two-dimensional tensor");
                    return;
                }
                std::string tensorType = "";
                if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::BF16) {
                    tensorType = "bfloat16";
                }
                else if (lhsDataType.sub_types[0].kind == PTO_TYPE_KIND::FP32) {
                    tensorType = "float32";
                }
                else {
                    SPDLOG_ERROR("Unsupported data type");
                    return;
                }
                create_device_memory_on_host(indent, fout, var->to_string(), lhsDataType.shape, tensorType);
            }
        }
        else {
            SPDLOG_ERROR("Unimplemented");
            return;
        }
        value->convert_to_triton_host(depth, fout);

        // 处理输出
        if (lhs->type() == PTO_NODE_TYPE::TYPED_VARIABLE) {
            if (kernelOutputPtr.find(kernelName) != kernelOutputPtr.end() && kernelOutputPtr[kernelName].find(0) != kernelOutputPtr[kernelName].end()) {
                fout << lhs->to_string() << ", " << lhs->to_string() << ".stride(0), " << lhs->to_string() << ".stride(1), ";
            }
        }
        else if (lhs->type() == PTO_NODE_TYPE::TUPLE_VARIABLE) {
            const auto& varList = ((PTO_TUPLE_VAR*)lhs)->get_var_list();
            for (std::size_t i = 0; i < varList.size(); ++i) {
                if (kernelOutputPtr.find(kernelName) != kernelOutputPtr.end() && kernelOutputPtr[kernelName].find(i) != kernelOutputPtr[kernelName].end()) {
                    fout << varList[i]->to_string() << ", " << varList[i]->to_string() << ".stride(0), " << varList[i]->to_string() << "stride(1), ";
                }
            }
        }
        else {
            SPDLOG_ERROR("Unimplemented");
            return;
        }

        fout << ")";
    }
    // 对于assemble需要特殊处理
    else if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.tensor.assemble") {
        // 这个函数特殊处理，如果第一个argument和lhs不是同一个变量，则需要拆成两个assignment
        const auto& args = ((PTO_CALL*)value)->get_arguments();

        if (args[0]->to_string() != lhs->to_string()) {
            lhs->dump_to_pyTorch(depth, fout);
            fout << " = ";
            args[0]->dump_to_pyTorch(0, fout);
            fout << std::endl;
        }
        
        lhs->dump_to_pyTorch(depth, fout);
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
        fout << indent;
        lhs->convert_to_triton_host(0, fout);
        fout << " = ";
        if (value->type() == PTO_NODE_TYPE::FUNC_CALL && ((PTO_CALL*)value)->get_func_name() == "pypto.language.tensor.create") {
            deviceMemoryPtr[lhs->to_string()] = std::vector<std::string>();
        }
        value->convert_to_triton_host(0, fout);
    }
}

void PTO_VARIABLE::convert_to_triton_host(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent << varName;
}

void PTO_CALL::convert_to_triton_host(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    fout << indent;

    if (funcName.substr(0, 5) == "self.") {
        fout << funcName.substr(5) << "[(";

        // 处理grid的大小
        if (programIDInfo.find(funcName.substr(5)) == programIDInfo.end()) {
            SPDLOG_ERROR("Unexpected Error");
            return;
        }
        const auto& curProgramIDInfo = programIDInfo[funcName.substr(5)];
        std::vector<int> grid(curProgramIDInfo.size(), 0);
        
        for (const auto& it : curProgramIDInfo) {
            if (it.second.range != -1) {
                grid[it.second.axis] = it.second.range;
            } else {
                if (it.second.rangePerCall.find(this) == it.second.rangePerCall.end()) {
                    SPDLOG_ERROR("Unexpected Error");
                    return;
                }
                grid[it.second.axis] = it.second.rangePerCall.find(this)->second;
            }
        }
        
        // 表明这个kernel函数没有被嵌入在任何一个parallel循环里
        if (grid.size() == 0) {
            grid.emplace_back(1);
        }

        for (const auto& g : grid) {
            fout << g << ", ";
        }
        fout << ")](";
        
        // 确定入参
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            if (curProgramIDInfo.find(i) != curProgramIDInfo.end()) {
                // 这个参量是通过program ID传入的
                continue;
            }
            fout << arguments[i]->to_string() << ", ";
            if (arguments[i]->get_data_type().kind == PTO_TYPE_KIND::TENSOR) {
                // 必须是二维
                if (arguments[i]->get_data_type().shape.size() != 2) {
                    SPDLOG_ERROR("Only support two-dimensional tensor");
                    return;
                }
                fout << arguments[i]->to_string() << ".stride(0), " << arguments[i]->to_string() << ".stride(1), ";
            }
        }

        // 输出结果的参数在调用这个函数的地方处理
    }
    else if (funcName == "pypto.language.tensor.create") {
        fout << "torch.empty(";
        arguments[0]->dump_to_pyTorch(0, fout);
        fout << ", ";
        arguments[1]->dump_to_pyTorch(0, fout);
        fout << ", ";
        arguments[2]->dump_to_pyTorch(0, fout);
        fout << ", ";
        fout << "device = 'cuda')";
    }
    else {
        this->dump_to_pyTorch(0, fout);
    }
}

void PTO_FOR_LOOP::convert_to_triton_host(int depth, std::ofstream& fout) const {
    std::string indent(depth, '\t');
    // 处理成串行的for loop
    fout << indent << "for ";
    iter->dump_to_pyTorch(0, fout);
    fout << " in range(";
    for (const auto& arg : info->get_arguments()) {
        if (arg->type() != PTO_NODE_TYPE::KEYWORD) {
            arg->dump_to_pyTorch(0, fout);
            fout << ", ";
        }
    }
    fout << "):" << std::endl;
    for (const auto& s : statements) {
        s->convert_to_triton_host(depth + 1, fout);
        fout << std::endl;
    }
}

void PTO_BINARY_OP::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // 直接调用torch的版本
    this->dump_to_pyTorch(depth, fout);
}

void PTO_RETURN::convert_to_triton_host(int depth, std::ofstream& fout) const {
    // 直接调用torch的版本
    this->dump_to_pyTorch(depth, fout);
}
    
} // namespace pto_parser
