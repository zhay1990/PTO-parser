#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

enum class VARIABLE_TYPE {
    INT32,
    TENSOR
};

enum class CALL_ARGUMENT_TYPE {
    VARIABLE,
    KEYWORD,
};

struct CALL_ARGUMENT {
    std::string varName;
    std::string value;

    CALL_ARGUMENT_TYPE type;
};

struct FUNCTION_CALL {
    std::string funcName;
    std::vector<struct CALL_ARGUMENT> arguments;

    void dump();
};

struct VARIABLE_NODE {
    std::string name;
    VARIABLE_TYPE type;


    struct FUNCTION_CALL call;
};

struct FUNCTION_NODE {

};

struct CLASS_NODE {
    std::string name;
    std::string decorator;

    std::vector<struct FUNCTION_NODE*> functions;

    int row, col;
};

struct MODULE_NODE {
    std::vector<struct CLASS_NODE*> classNodes;
};