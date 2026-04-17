#include "parser.hh"
#include "logger.hh"
#include <fstream>
#include <tree_sitter/api.h>
#include <string.h>
#include <string>


// 辅助函数
static const std::string get_node_text(TSNode node, const std::string& buffer) {
    if (ts_node_is_null(node)) return "";

    uint32_t start = ts_node_start_byte(node);
    uint32_t end   = ts_node_end_byte(node);

    return buffer.substr(start, end - start);
}

static inline bool check_node_type(TSNode node, const std::string& name) {
    if (ts_node_is_null(node)) return false;
    return strcmp(ts_node_type(node), name.c_str()) == 0;
}

static inline bool check_node_type(TSNode node, const uint32_t& childIndex, const std::string& name) {
    if (ts_node_is_null(node)) return false;
    if (ts_node_named_child_count(node) <= childIndex) return false;
    return strcmp(ts_node_type(ts_node_named_child(node, childIndex)), name.c_str()) == 0;
}


static inline const std::string get_attribute_str(TSNode node, const std::string& buffer) {
    if (!check_node_type(node, "attribute")) {
        SPDLOG_ERROR("get_attribute_str only used for attribute node");
        return "";
    }

    std::string ret;

    if (check_node_type(ts_node_named_child(node, 0), "attribute")) {
        ret = get_attribute_str(ts_node_named_child(node, 0), buffer);
    }
    else if (check_node_type(ts_node_named_child(node, 0), "identifier")) {
        ret = get_node_text(ts_node_named_child(node, 0), buffer);
    }
    else {
        SPDLOG_ERROR("Unexpected node type {}", ts_node_type(node));
        return "";
    }

    for (uint32_t i = 1; i < ts_node_named_child_count(node); ++i) {
        ret += "." + get_node_text(ts_node_named_child(node, i), buffer);
    }

    return ret;
}

static inline const std::string replace_import_alias(const std::string& in, const pto_parser::STR_STR_MAP& importMap) {
    size_t index = in.find_first_of('.');
    if (index == std::string::npos) return in;
    std::string temp = in.substr(0, index);
    if (importMap.find(temp) == importMap.end()) return in;
    return importMap.find(temp)->second + in.substr(index);
}

// 为了是SPDLOG_ERROR能捕捉到错误的行号，这个函数被定义成宏
#define UNIMPLEMENTED_ERROR(node, buffer)                                         \
    do {                                                                          \
        /* 使用局部变量缓存，防止传入含有副作用的表达式被多次求值 */                     \
        auto _mac_node = (node);                                                  \
        if (!ts_node_is_null(_mac_node)) {                                        \
            uint32_t _mac_start = ts_node_start_byte(_mac_node);                  \
            uint32_t _mac_end = ts_node_end_byte(_mac_node);                      \
            TSPoint _mac_point = ts_node_start_point(_mac_node);                  \
                                                                                  \
            std::string _mac_text = (buffer).substr(_mac_start, _mac_end - _mac_start); \
            std::replace(_mac_text.begin(), _mac_text.end(), '\n', ' ');          \
            if (_mac_text.length() > 40) {                                        \
                _mac_text = _mac_text.substr(0, 37) + "...";                      \
            }                                                                     \
                                                                                  \
            SPDLOG_ERROR("Process method for '{}' at line {} is not implemented.", \
                         _mac_text, _mac_point.row + 1);                          \
        }                                                                         \
    } while (0)

extern "C" const TSLanguage* tree_sitter_python();

static pto_parser::PTO_CALL* create_call_node(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias);

static void dump_tree_for_debug(TSNode node, const std::string& source, const int& depth) {
    if (ts_node_is_null(node)) return;

    const char* type = ts_node_type(node);
    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);
    TSPoint point = ts_node_start_point(node);

    // 提取并清理文本（去除换行符，限制长度）
    std::string text = source.substr(start, end - start);
    std::replace(text.begin(), text.end(), '\n', ' ');
    if (text.length() > 40) text = text.substr(0, 37) + "...";

    std::string indent(depth * 2, ' ');
    
    // 打印到日志文件
    SPDLOG_DEBUG("{}[{}] L{}: \"{}\"", indent, type, point.row + 1, text);

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        dump_tree_for_debug(ts_node_child(node, i), source, depth + 1);
    }
}

static void handle_import(TSNode node, const std::string& buffer, pto_parser::STR_STR_MAP& importAlias) {
    // 分情况处理
    // 1. import xxx.xx as xx
    // 这种模式下node只有一个aliased_import节点
    if (ts_node_named_child_count(node) == 1 && check_node_type(ts_node_named_child(node, 0), "aliased_import")) {
        TSNode child = ts_node_named_child(node, 0);
        if (ts_node_named_child_count(child) != 2) {
            SPDLOG_ERROR("Unexpected error for aliased import");
            return;
        }

        std::string alias = get_node_text(ts_node_named_child(child, 1), buffer);
        std::string original = get_node_text(ts_node_named_child(child, 0), buffer);

        if (importAlias.find(alias) != importAlias.end()) {
            SPDLOG_WARN("Duplicated alias '{}' for '{}' and '{}'", alias, original, importAlias[alias]);
        }

        importAlias[alias] = original;
        SPDLOG_DEBUG("Got alias import name '{}' for '{}'", alias, original);
        
        return;
    }
    // 2. import xxx.xx
    if (ts_node_named_child_count(node) == 1 && check_node_type(ts_node_named_child(node, 0), "dotted_name")) {
        // 不需要处理任何事情
        return;
    }

    // 其他模式待补充
    UNIMPLEMENTED_ERROR(node, buffer);
}

static pto_parser::PTO_BINARY_OP* parse_comparison_operator(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP&) {
    if (!check_node_type(node, "comparison_operator")) {
        SPDLOG_ERROR("parse_comparison_operator only used for comparison_operator node");
        return nullptr;
    }

    if (ts_node_child_count(node) != 3) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }

    pto_parser::PTO_BINARY_OP *ret = nullptr;

    // 根据第二个child确定操作类型
    if (check_node_type(ts_node_child(node, 1), "==")) {
        ret = new pto_parser::PTO_BINARY_OP(pto_parser::PTO_OPERATOR::EQUAL, ts_node_start_point(node).row + 1, ts_node_start_point(node).column);
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_child(node, 1), buffer);
        return ret;
    }

    // 处理lhs，目前看到以下几种情况
    if (check_node_type(node, 0, "identifier")) {
        ret->set_lhs(new pto_parser::PTO_VARIABLE(
            get_node_text(ts_node_named_child(node, 0), buffer),
            ts_node_start_point(ts_node_named_child(node, 0)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 0)).column
        ));
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_named_child(node, 0), buffer);
    }

    // 处理rhs，目前看到以下几种情况
    if (check_node_type(node, 1, "integer")) {
        ret->set_rhs(new pto_parser::PTO_INT(
            std::stoi(get_node_text(ts_node_named_child(node, 1), buffer)),
            ts_node_start_point(ts_node_named_child(node, 1)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 1)).column
        ));
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_named_child(node, 1), buffer);
    }

    return ret;
}

static pto_parser::PTO_BINARY_OP* parse_binary_operator(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "binary_operator")) {
        SPDLOG_ERROR("parse_binary_operator only used for binary_operator node");
        return nullptr;
    }

    if (ts_node_child_count(node) != 3) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }

    pto_parser::PTO_BINARY_OP *ret = nullptr;

    // 根据第二个child确定操作类型
    if (check_node_type(ts_node_child(node, 1), "*")) {
        ret = new pto_parser::PTO_BINARY_OP(pto_parser::PTO_OPERATOR::MUL, ts_node_start_point(node).row + 1, ts_node_start_point(node).column);
    }
    else if (check_node_type(ts_node_child(node, 1), "-")) {
        ret = new pto_parser::PTO_BINARY_OP(pto_parser::PTO_OPERATOR::SUB, ts_node_start_point(node).row + 1, ts_node_start_point(node).column);
    }
    else if (check_node_type(ts_node_child(node, 1), "//")) {
        ret = new pto_parser::PTO_BINARY_OP(pto_parser::PTO_OPERATOR::FLOOR_DIV, ts_node_start_point(node).row + 1, ts_node_start_point(node).column);
    }
    else if (check_node_type(ts_node_child(node, 1), "+")) {
        ret = new pto_parser::PTO_BINARY_OP(pto_parser::PTO_OPERATOR::ADD, ts_node_start_point(node).row + 1, ts_node_start_point(node).column);
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_child(node, 1), buffer);
        return ret;
    }

    // 处理lhs，目前看到以下几种情况
    if (check_node_type(node, 0, "identifier")) {
        ret->set_lhs(new pto_parser::PTO_VARIABLE(
            get_node_text(ts_node_named_child(node, 0), buffer),
            ts_node_start_point(ts_node_named_child(node, 0)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 0)).column
        ));
    }
    else if (check_node_type(node, 0, "parenthesized_expression")) {
        if (ts_node_named_child_count(ts_node_named_child(node, 0)) == 1 && check_node_type(ts_node_named_child(node, 0), 0, "binary_operator")) {
            ret->set_lhs(parse_binary_operator(ts_node_named_child(ts_node_named_child(node, 0), 0), buffer, importAlias));
        }
        else {
            UNIMPLEMENTED_ERROR(ts_node_named_child(ts_node_named_child(node, 0), 0), buffer);
            return ret;
        }
        
    }
    else if (check_node_type(node, 0, "binary_operator")) {
        ret->set_lhs(parse_binary_operator(ts_node_named_child(node, 0), buffer, importAlias));
    }
    else if (check_node_type(node, 0, "call")) {
        ret->set_lhs(create_call_node(ts_node_named_child(node, 0), buffer, importAlias));
    }
    else if (check_node_type(node, 0, "integer")) {
        ret->set_lhs(new pto_parser::PTO_INT(
            std::stoi(get_node_text(ts_node_named_child(node, 0), buffer)),
            ts_node_start_point(ts_node_named_child(node, 0)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 0)).column
        ));
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_named_child(node, 0), buffer);
    }

    // 处理rhs，目前看到以下几种情况
    if (check_node_type(node, 1, "identifier")) {
        ret->set_rhs(new pto_parser::PTO_VARIABLE(
            get_node_text(ts_node_named_child(node, 1), buffer),
            ts_node_start_point(ts_node_named_child(node, 1)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 1)).column
        ));
    }
    else if (check_node_type(node, 1, "binary_operator")) {
        ret->set_rhs(parse_binary_operator(ts_node_named_child(node, 1), buffer, importAlias));
    }
    else if (check_node_type(node, 1, "integer")) {
        ret->set_rhs(new pto_parser::PTO_INT(
            std::stoi(get_node_text(ts_node_named_child(node, 1), buffer)),
            ts_node_start_point(ts_node_named_child(node, 1)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 1)).column
        ));
    }
    else if (check_node_type(node, 1, "parenthesized_expression")) {
        TSNode expr = ts_node_named_child(node, 1);
        if (ts_node_named_child_count(expr) != 1) {
            SPDLOG_ERROR("Unexpected Error");
        }
        else if (check_node_type(expr, 0, "binary_operator")) {
            ret->set_rhs(parse_binary_operator(ts_node_named_child(expr, 0), buffer, importAlias));
        }
        else {
            UNIMPLEMENTED_ERROR(ts_node_named_child(expr, 0), buffer);    
        }
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_named_child(node, 1), buffer);
    }

    return ret;
}

static pto_parser::PTO_LIST_VAR* create_list_var(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP&) {
    if (!check_node_type(node, "list")) {
        SPDLOG_ERROR("create_list_var only used for list node");
        return nullptr;
    }

    auto ret = new pto_parser::PTO_LIST_VAR(ts_node_start_point(node).row + 1, ts_node_start_point(node).column);

    for (std::size_t i = 0; i < ts_node_named_child_count(node); ++i) {
        TSNode child = ts_node_named_child(node, i);

        if (check_node_type(child, "identifier")) {
            ret->add_var(new pto_parser::PTO_VARIABLE(
                get_node_text(child, buffer),
                ts_node_start_point(child).row + 1,
                ts_node_start_point(child).column
            ));
        }
        else if (check_node_type(child, "integer")) {
            ret->add_var(new pto_parser::PTO_INT(
                std::stoi(get_node_text(child, buffer)),
                ts_node_start_point(child).row + 1,
                ts_node_start_point(child).column
            ));
        }
        else {
            UNIMPLEMENTED_ERROR(node, buffer);
        }
    }

    return ret;
}

static pto_parser::PTO_CALL* create_call_node(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "call")) {
        SPDLOG_ERROR("create_call_node only used for call node");
        return nullptr;
    }

    // 应当有两个命名节点
    if (ts_node_named_child_count(node) != 2) {
        SPDLOG_ERROR("Unexpected error");
        return nullptr;
    }

    // 第一个节点是funcName
    std::string funcName = get_node_text(ts_node_named_child(node, 0), buffer);
    funcName = replace_import_alias(funcName, importAlias);

    // 第二个节点是argument list
    if (!check_node_type(ts_node_named_child(node, 1), "argument_list")) {
        SPDLOG_ERROR("Unexpected error");
        return nullptr;
    }

    TSNode argumentList = ts_node_named_child(node, 1);

    std::vector<pto_parser::PTO_BASE*> arguments;

    for (uint32_t i = 0; i < ts_node_named_child_count(argumentList); ++i) {
        TSNode param = ts_node_named_child(argumentList, i);

        if (check_node_type(param, "string") || check_node_type(param, "identifier")) {
            pto_parser::PTO_VARIABLE *variable = new pto_parser::PTO_VARIABLE(
                replace_import_alias(get_node_text(param, buffer), importAlias),
                ts_node_start_point(param).row + 1,
                ts_node_start_point(param).column + 1
            );

            arguments.emplace_back(variable);
        }
        else if (check_node_type(param, "attribute")) {
            std::string varName = get_attribute_str(param, buffer);
            varName = replace_import_alias(varName, importAlias);
            pto_parser::PTO_VARIABLE *variable = new pto_parser::PTO_VARIABLE(
                varName,
                ts_node_start_point(param).row + 1,
                ts_node_start_point(param).column + 1
            );

            arguments.emplace_back(variable);
        }
        else if (check_node_type(param, "keyword_argument")) {
            TSNode var = ts_node_named_child(param, 0);
            TSNode val = ts_node_named_child(param, 1);

            pto_parser::PTO_KEYWORD *variable = new pto_parser::PTO_KEYWORD(
                get_node_text(var, buffer),
                ts_node_start_point(var).row + 1,
                ts_node_start_point(var).column + 1
            );

            if (check_node_type(val, "identifier") || check_node_type(val, "attribute") || check_node_type(val, "string")) {
                variable->set_value(new pto_parser::PTO_VARIABLE(
                    replace_import_alias(get_node_text(val, buffer), importAlias),
                    ts_node_start_point(val).row + 1,
                    ts_node_start_point(val).column
                ));
            }
            else if (check_node_type(val, "tuple") || check_node_type(val, "parenthesized_expression")) {
                auto valPtr = new pto_parser::PTO_TUPLE_VAR(
                    ts_node_start_point(val).row + 1,
                    ts_node_start_point(val).column
                );
                for (uint32_t j = 0; j < ts_node_named_child_count(val); ++j) {
                    if (check_node_type(val, j, "identifier")) {
                        valPtr->add_var(new pto_parser::PTO_VARIABLE(
                            get_node_text(ts_node_named_child(val, j), buffer),
                            ts_node_start_point(ts_node_named_child(val, j)).row + 1,
                            ts_node_start_point(ts_node_named_child(val, j)).column
                        ));
                    }
                    else {
                        UNIMPLEMENTED_ERROR(val, buffer);
                    }
                }
                variable->set_value(valPtr);
            }
            else if (check_node_type(val, "integer")) {
                variable->set_value(new pto_parser::PTO_INT(
                    std::stoi(get_node_text(val, buffer)),
                    ts_node_start_point(val).row + 1,
                    ts_node_start_point(val).column
                ));
            } 
            else if (check_node_type(val, "false")) {
                variable->set_value(new pto_parser::PTO_BOOL(
                    false,
                    ts_node_start_point(val).row + 1,
                    ts_node_start_point(val).column
                ));
            }
            else if (check_node_type(val, "true")) {
                variable->set_value(new pto_parser::PTO_BOOL(
                    true,
                    ts_node_start_point(val).row + 1,
                    ts_node_start_point(val).column
                ));
            }
            else if (check_node_type(val, "list")) {
                variable->set_value(create_list_var(val, buffer, importAlias));
            }
            else {
                UNIMPLEMENTED_ERROR(node, buffer);
            }

            arguments.emplace_back(variable);
        }
        else if (check_node_type(param, "float")) {
            arguments.emplace_back(new pto_parser::PTO_FLOAT(
                std::stof(get_node_text(param, buffer)),
                ts_node_start_point(param).row + 1,
                ts_node_start_point(param).column
            ));
        } else if (check_node_type(param, "integer")) {
            arguments.emplace_back(new pto_parser::PTO_INT(
                std::stoi(get_node_text(param, buffer)),
                ts_node_start_point(param).row + 1,
                ts_node_start_point(param).column
            ));
        }
        else if (check_node_type(param, "list")) {
            if (funcName == "pypto.language.Tuple") {
                // 在typed variable类型定义中出现，将list中的每个参数都展开

                // 解析类型，会有多个child，每个child都应当是subcript类型
                for (uint32_t j = 0; j < ts_node_named_child_count(param); ++j) {
                    TSNode type2 = ts_node_named_child(param, j);
                    if (!check_node_type(type2, "subscript")) {
                        SPDLOG_ERROR("Unexpected ERROR");
                        continue;
                    }

                    std::string typeStr;
                    for (uint32_t k = 0; k < ts_node_child_count(type2); ++k) {
                        if (check_node_type(ts_node_child(type2, k), "attribute")) {
                            std::string temp = get_node_text(ts_node_child(type2, k), buffer);
                            typeStr += replace_import_alias(temp, importAlias);
                        } else {
                            typeStr += get_node_text(ts_node_child(type2, k), buffer);
                        }
                    }
                    arguments.emplace_back(new pto_parser::PTO_VARIABLE(typeStr, 0, 0));
                }
            } else {
                // 作为一个参数处理
                arguments.emplace_back(create_list_var(param, buffer, importAlias));
            }

        }
        else if (check_node_type(param, "parenthesized_expression")) {
            // 应当只有一个child
            if (ts_node_named_child_count(param) != 1) {
                SPDLOG_ERROR("Unexpected Error");
            }
            else if (check_node_type(param, 0, "binary_operator")) {
                arguments.emplace_back(parse_binary_operator(ts_node_named_child(param, 0), buffer, importAlias));
            }
            else {
                UNIMPLEMENTED_ERROR(ts_node_named_child(param, 0), buffer);
            }
        }
        else if (check_node_type(param, "binary_operator")) {
            arguments.emplace_back(parse_binary_operator(param, buffer, importAlias));
        }
        else {
            UNIMPLEMENTED_ERROR(node, buffer);
        }
    }

    pto_parser::PTO_CALL *ret = new pto_parser::PTO_CALL(
        funcName,
        ts_node_start_point(node).row + 1,
        ts_node_start_point(node).column + 1
    );

    ret->add_arguments(arguments);

    return ret;
}

// 这个函数是为全局变量定义写的，后续需要改名，有歧义
static pto_parser::PTO_ASSIGNMENT* create_assignment(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "expression_statement")) {
        SPDLOG_ERROR("create_assignment only used for expression_statement node");
        return nullptr;
    }

    // 处理只有一个assignment节点的情况
    if (ts_node_named_child_count(node) == 1 && check_node_type(ts_node_named_child(node, 0), "assignment")) {
        TSNode assign = ts_node_named_child(node, 0);

        // 应当只有两个命名节点，一个是左值一个是右值
        if (ts_node_named_child_count(assign) != 2) {
            SPDLOG_ERROR("Unexpected error for assignment at line {}", ts_node_start_point(assign).row + 1);
            return nullptr;
        }

        // 对于左值的处理
        pto_parser::PTO_BASE *lhs = nullptr;
        if (check_node_type(assign, 0, "pattern_list") || check_node_type(assign, 0, "tuple_pattern")) {
            // 处理为tuple var
            TSNode list = ts_node_named_child(assign, 0);

            auto lhs_temp = new pto_parser::PTO_TUPLE_VAR(
                ts_node_start_point(list).row + 1,
                ts_node_start_point(list).column
            );

            for (uint32_t i = 0; i < ts_node_named_child_count(list); ++i) {
                TSNode var = ts_node_named_child(list, i);
                if (!check_node_type(var, "identifier")) {
                    SPDLOG_ERROR("Unexpected Error");
                    continue;
                }

                lhs_temp->add_var(new pto_parser::PTO_VARIABLE(
                    get_node_text(var, buffer),
                    ts_node_start_point(var).row + 1,
                    ts_node_start_point(var).column
                ));
            }

            lhs = lhs_temp;
        }
        else if (check_node_type(assign, 0, "identifier")) {
            lhs = new pto_parser::PTO_VARIABLE(
                get_node_text(ts_node_named_child(assign, 0), buffer),
                ts_node_start_point(ts_node_named_child(assign, 0)).row + 1,
                ts_node_start_point(ts_node_named_child(assign, 0)).column
            );
        }
        else {
            SPDLOG_ERROR("Process method for {} in assignment at line {} is not implemented", ts_node_type(ts_node_named_child(assign, 0)), ts_node_start_point(node).row + 1);
            return nullptr;
        }

        // 处理右值
        if (check_node_type(ts_node_named_child(assign, 1), "call")) {
            pto_parser::PTO_CALL *callNode = create_call_node(ts_node_named_child(assign, 1), buffer, importAlias);
            
            pto_parser::PTO_ASSIGNMENT* ret = new pto_parser::PTO_ASSIGNMENT(
                lhs,
                ts_node_start_point(node).row + 1,
                ts_node_start_point(node).column + 1
            );

            ret->set_value(callNode);

            return ret;
        }
    }
    UNIMPLEMENTED_ERROR(node, buffer);
    return nullptr;
}

static const std::string get_decorate_string(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "decorator")) {
        SPDLOG_ERROR("get_decorate_string only process decorate node");
        return "";
    }

    std::string decorate = "";

    // 现在看到了三种情况
    if (ts_node_named_child_count(node) == 1) {
        if (check_node_type(ts_node_named_child(node, 0), "attribute")) {
            decorate = get_attribute_str(ts_node_named_child(node, 0), buffer);
            decorate = replace_import_alias(decorate, importAlias);
        }
        else if (check_node_type(ts_node_named_child(node, 0), "identifier")) {
            decorate = get_node_text(ts_node_named_child(node, 0), buffer);
        }
        else if (check_node_type(ts_node_named_child(node, 0), "call")) {
            auto ptr = create_call_node(ts_node_named_child(node, 0), buffer, importAlias);
            decorate = ptr->to_string();
            delete ptr;
        }
        else {
            UNIMPLEMENTED_ERROR(node, buffer);
        }
    }

    return decorate;
}

pto_parser::PTO_VARIABLE* create_type_variable(TSNode identifier, TSNode type, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(identifier, "identifier")) {
        SPDLOG_ERROR("The first node for create_type_variable should be 'identifier'");
        return nullptr;
    }
    if (!check_node_type(type, "type")) {
        SPDLOG_ERROR("The second node for create_type_variable should be 'type'");
        return nullptr;
    }

    auto ret = new pto_parser::PTO_VARIABLE(
        get_node_text(identifier, buffer),
        ts_node_start_point(identifier).row + 1,
        ts_node_start_point(identifier).column
    );

    // 只处理只有一个节点的type
    if (ts_node_named_child_count(type) != 1) {
        SPDLOG_ERROR("The number of children for '{}' is more than 1", get_node_text(type, buffer));
    }

    // 当前看到两种情况
    if (check_node_type(type, 0, "subscript")) {
        std::string typeStr;
        TSNode subscript = ts_node_named_child(type, 0);

        for (uint32_t j = 0; j < ts_node_child_count(subscript); ++j) {
            TSNode child = ts_node_child(subscript, j);
            if (check_node_type(child, "attribute")) {
                std::string temp = get_node_text(child, buffer);
                typeStr += replace_import_alias(temp, importAlias);
            }
            else if (check_node_type(child, "call")) {
                auto func = create_call_node(child, buffer, importAlias);
                if (func == nullptr) {
                    SPDLOG_ERROR("Unexpected error");
                }
                typeStr += func->get_func_name() + "(";
                
                if (func->get_arguments().size() != 0) {
                    typeStr += func->get_arguments()[0]->to_string();
                }
                for (std::size_t k = 1; k < func->get_arguments().size(); ++k) {
                    typeStr += ", " + func->get_arguments()[k]->to_string();
                }
                typeStr += ")";
                delete func;
            }
            else {
                typeStr += get_node_text(child, buffer);
            }
        }

        // 将string存入节点 后续解析是什么类型
        ret->add_type_str(typeStr);
    }
    else if (check_node_type(type, 0, "call")) {
        // 复用create_call_node方法解析
        auto func = create_call_node(ts_node_named_child(type, 0), buffer, importAlias);
        
        // 类型当前只支持tuple
        if (func->get_func_name() != "pypto.language.Tuple") {
            UNIMPLEMENTED_ERROR(type, buffer);
        }

        // 拿到argument名字
        for (const auto& arg : func->get_arguments()) {
            ret->add_type_str(arg->to_string());
        }
        
        delete func;
    }
    else {
        UNIMPLEMENTED_ERROR(type, buffer);
    }

    return ret;
}

static pto_parser::PTO_ASSIGNMENT* create_typed_assignment(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "assignment")) {
        SPDLOG_ERROR("create_typed_assignment can only used for assignment node");
        return nullptr;
    }
    
    // 只处理有三个子节点的case
    if (ts_node_named_child_count(node) != 3 || !check_node_type(node, 0, "identifier") || !check_node_type(node, 1, "type")) {
        SPDLOG_ERROR("Unexpected error");
        return nullptr;
    }

    auto ptr = create_type_variable(
        ts_node_named_child(node, 0),
        ts_node_named_child(node, 1),
        buffer,
        importAlias
    );

    auto assignNode = new pto_parser::PTO_ASSIGNMENT(
        ptr,
        ts_node_start_point(node).row + 1,
        ts_node_start_point(node).column
    );

    // 处理第三个节点
    if (check_node_type(node, 2, "call")) {
        auto call = create_call_node(ts_node_named_child(node, 2), buffer, importAlias);
        assignNode->set_value(call);
    }
    else if (check_node_type(node, 2, "subscript")) {
        // 类似于ret[0]类型，需要记录index
        // 先只处理只有两个节点的case
        if (ts_node_named_child_count(ts_node_named_child(node, 2)) != 2 || !check_node_type(ts_node_named_child(node, 2), 0, "identifier") || !check_node_type(ts_node_named_child(node, 2), 1, "integer")) {
            UNIMPLEMENTED_ERROR(ts_node_named_child(node, 2), buffer);
        }
        assignNode->set_value(new pto_parser::PTO_INDEXED_VAR(
            get_node_text(ts_node_named_child(ts_node_named_child(node, 2), 0), buffer),
            std::stoi(get_node_text(ts_node_named_child(ts_node_named_child(node, 2), 1), buffer)),
            ts_node_start_point(ts_node_named_child(node, 2)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 2)).column
        ));
    }
    else if (check_node_type(node, 2, "binary_operator")) {
        auto ptr = parse_binary_operator(ts_node_named_child(node, 2), buffer, importAlias);
        assignNode->set_value(ptr);
    }
    else if (check_node_type(node, 2, "identifier")) {
        assignNode->set_value(new pto_parser::PTO_VARIABLE(
            get_node_text(ts_node_named_child(node, 2), buffer),
            ts_node_start_point(ts_node_named_child(node, 2)).row + 1,
            ts_node_start_point(ts_node_named_child(node, 2)).column
        ));
    } else if (check_node_type(node, 2, "parenthesized_expression")) {
        // 应当只有一个named child
        TSNode expression = ts_node_named_child(node, 2);
        if (ts_node_named_child_count(expression) != 1) {
            SPDLOG_ERROR("Unexpected Error");
            return assignNode;
        }

        if (check_node_type(expression, 0, "binary_operator")) {
            assignNode->set_value(parse_binary_operator(ts_node_named_child(expression, 0), buffer, importAlias));
        }
        else {
            UNIMPLEMENTED_ERROR(ts_node_named_child(expression, 0), buffer);    
        }
    }
    else {
        UNIMPLEMENTED_ERROR(ts_node_named_child(node, 2), buffer);
    }

    return assignNode;
    
}

static std::vector<pto_parser::PTO_BASE*> parse_block_node(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    auto ret = std::vector<pto_parser::PTO_BASE*>();

    if (!check_node_type(node, "block")) {
        SPDLOG_ERROR("create_block_node only used for block node");
        return ret;
    }

    for (uint32_t i = 0; i < ts_node_named_child_count(node); ++i) {
        TSNode statement = ts_node_named_child(node, i);

        if (check_node_type(statement, "expression_statement")) {
            if (ts_node_named_child_count(statement) == 1 && check_node_type(ts_node_named_child(statement, 0), "assignment")) {
                TSNode assignment = ts_node_named_child(statement, 0);
                
                if (ts_node_named_child_count(assignment) == 3) {
                    auto ptr = create_typed_assignment(assignment, buffer, importAlias);
                    if (ptr != nullptr) {
                        ret.emplace_back(ptr);
                        continue;
                    }
                }
                else {
                    auto ptr = create_assignment(statement, buffer, importAlias);
                    if (ptr != nullptr) {
                        ret.emplace_back(ptr);
                        continue;
                    }
                }
            }
        }
        else if (check_node_type(statement, "for_statement")) {
            if (ts_node_named_child_count(statement) == 3 && check_node_type(statement, 0, "pattern_list") && check_node_type(statement, 1, "call") && check_node_type(statement, 2, "block")) {
                auto pattern_list = ts_node_named_child(statement, 0);
                auto call = ts_node_named_child(statement, 1);
                auto block = ts_node_named_child(statement, 2);
                
                auto ptr = new pto_parser::PTO_FOR_LOOP(ts_node_start_point(statement).row + 1, ts_node_start_point(statement).column);

                // 对于第一个pattern_list，只支持有两个节点的情况
                if (ts_node_named_child_count(pattern_list) != 2 || !check_node_type(pattern_list, 0, "identifier") || !check_node_type(pattern_list, 1, "tuple_pattern")) {
                    UNIMPLEMENTED_ERROR(pattern_list, buffer);
                } else {
                    // 解析循环变量
                    ptr->set_iterator(new pto_parser::PTO_VARIABLE(
                        get_node_text(ts_node_named_child(pattern_list, 0), buffer),
                        ts_node_start_point(ts_node_named_child(pattern_list, 0)).row + 1,
                        ts_node_start_point(ts_node_named_child(pattern_list, 0)).column
                    ));

                    // 解析初始变量
                    for (unsigned int j = 0; j < ts_node_named_child_count(ts_node_named_child(pattern_list, 1)); ++j) {
                        auto id = ts_node_named_child(ts_node_named_child(pattern_list, 1), j);
                        // 只处理identifier的情况
                        if (check_node_type(id, "identifier")) {
                            ptr->add_init_variable(new pto_parser::PTO_VARIABLE(
                                get_node_text(id, buffer),
                                ts_node_start_point(id).row + 1,
                                ts_node_start_point(id).column
                            ));
                        }
                        else {
                            UNIMPLEMENTED_ERROR(id, buffer);
                        }
                    }
                }

                // 解析call
                ptr->set_call_info(create_call_node(call, buffer, importAlias));

                // 解析block信息
                ptr->add_statement(parse_block_node(block, buffer, importAlias));
                
                ret.emplace_back(ptr);
                continue;
            }
        }
        else if (check_node_type(statement, "if_statement")) {
            auto ptr = new pto_parser::PTO_IF(ts_node_start_point(statement).row + 1, ts_node_start_point(statement).column);
            // 第一个child是comparator
            // 可能会套多层parenthesized_expression
            TSNode comp = ts_node_named_child(statement, 0);
            while (!ts_node_is_null(comp)) {
                if (check_node_type(comp, "comparison_operator")) {
                    break;
                }

                if (!check_node_type(comp, "parenthesized_expression")) {
                    SPDLOG_ERROR("Unexpected Error");
                    break;
                }

                if (ts_node_named_child_count(comp) != 1) {
                    SPDLOG_ERROR("Unexpected Error");
                    break;
                }

                comp = ts_node_named_child(comp, 0);
            }

            ptr->set_comparator(parse_comparison_operator(comp, buffer, importAlias));

            // 第二个child是block
            ptr->add_if_statements(parse_block_node(ts_node_named_child(statement, 1), buffer, importAlias));

            // 处理第三个child是else_clause的情况
            if (!check_node_type(statement, 2, "else_clause")) {
                UNIMPLEMENTED_ERROR(ts_node_named_child(statement, 2), buffer);
            } else {
                ptr->add_else_statements(parse_block_node(ts_node_named_child(ts_node_named_child(statement, 2), 0), buffer, importAlias));
            }
            
            ret.emplace_back(ptr);
            continue;
        }
        else if (check_node_type(statement, "return_statement")) {
            // 只处理两种情况
            if (ts_node_named_child_count(statement) == 1 && check_node_type(statement, 0, "expression_list")) {
                TSNode expressionList = ts_node_named_child(statement, 0);
                auto ptr = new pto_parser::PTO_RETURN(ts_node_start_point(statement).row + 1, ts_node_start_point(statement).column);
                for (uint32_t j = 0; j < ts_node_named_child_count(expressionList); ++j) {
                    if (check_node_type(ts_node_named_child(expressionList, j), "identifier")) {
                        ptr->add_value(new pto_parser::PTO_VARIABLE(
                            get_node_text(ts_node_named_child(expressionList, j), buffer),
                            ts_node_start_point(ts_node_named_child(expressionList, j)).row + 1,
                            ts_node_start_point(ts_node_named_child(expressionList, j)).column
                        ));
                    }
                    else {
                        UNIMPLEMENTED_ERROR(statement, buffer);
                    }
                }

                ret.emplace_back(ptr);
                continue;
            } 
            else if (ts_node_named_child_count(statement) == 1 && check_node_type(statement, 0, "identifier")) {
                // 返回一个identifier的情况
                auto ptr = new pto_parser::PTO_RETURN(ts_node_start_point(statement).row + 1, ts_node_start_point(statement).column);
                
                ptr->add_value(new pto_parser::PTO_VARIABLE(
                    get_node_text(ts_node_named_child(statement, 0), buffer),
                    ts_node_start_point(ts_node_named_child(statement, 0)).row + 1,
                    ts_node_start_point(ts_node_named_child(statement, 0)).column
                ));
                
                ret.emplace_back(ptr);
                continue;
            }
        }
        
        UNIMPLEMENTED_ERROR(statement, buffer);
    }

    return ret;
}

static pto_parser::PTO_FUNC* create_func_node(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "function_definition")) {
        SPDLOG_ERROR("create_func_node can only used for function_definition node");
        return nullptr;
    }

    // 第一个节点应当是identifier
    if (!check_node_type(ts_node_named_child(node, 0), "identifier")) {
        SPDLOG_ERROR("Unexpected error");
        return nullptr;
    }

    pto_parser::PTO_FUNC *ret = new pto_parser::PTO_FUNC(
        get_node_text(ts_node_named_child(node, 0), buffer),
        ts_node_start_point(node).row + 1,
        ts_node_start_point(node).column
    );

    // 第二个节点是parameters
    if (!check_node_type(ts_node_named_child(node, 1), "parameters")) {
        SPDLOG_ERROR("Unexpected error");
        return nullptr;
    }

    std::vector<pto_parser::PTO_VARIABLE*> arguments;

    for (uint32_t i = 0; i < ts_node_named_child_count(ts_node_named_child(node, 1)); ++i) {
        TSNode param = ts_node_named_child(ts_node_named_child(node, 1), i);

        // 看到两种情况
        if (check_node_type(param, "identifier")) {
            arguments.emplace_back(new pto_parser::PTO_VARIABLE(
                get_node_text(param, buffer),
                ts_node_start_point(param).row + 1,
                ts_node_start_point(param).column
            ));
        }
        else if (check_node_type(param, "typed_parameter")) {
            arguments.emplace_back(new pto_parser::PTO_VARIABLE(
                get_node_text(ts_node_named_child(param, 0), buffer),
                ts_node_start_point(ts_node_named_child(param, 0)).row + 1,
                ts_node_start_point(ts_node_named_child(param, 0)).column
            ));

            // 解析类型
            TSNode type = ts_node_named_child(ts_node_named_child(param, 1), 0);
            if (!check_node_type(type, "subscript")) {
                SPDLOG_ERROR("Unexpected ERROR");
                continue;
            }

            std::string typeStr;
            for (uint32_t j = 0; j < ts_node_child_count(type); ++j) {
                if (check_node_type(ts_node_child(type, j), "attribute")) {
                    std::string temp = get_node_text(ts_node_child(type, j), buffer);
                    typeStr += replace_import_alias(temp, importAlias);
                } else {
                    typeStr += get_node_text(ts_node_child(type, j), buffer);
                }
            }

            // 将string存入节点 后续解析是什么类型
            arguments.back()->add_type_str(typeStr);
        }
        else {
            UNIMPLEMENTED_ERROR(param, buffer);
        }
    }

    ret->add_arguments(arguments);

    // 第三节点可能是type, 表示函数的返回值
    if (check_node_type(ts_node_named_child(node, 2), "type")) {
        TSNode type = ts_node_named_child(node, 2);
        // 当前只支持单Variable和tuple类型的返回值
        std::vector<std::string> returnTypes;
        if (ts_node_named_child_count(type) == 1 && 
            check_node_type(ts_node_named_child(type, 0), "generic_type") && 
            get_node_text(ts_node_named_child(ts_node_named_child(type, 0), 0), buffer) == "tuple") {
            
            TSNode params = ts_node_named_child(ts_node_named_child(type, 0), 1);
            
            for (uint32_t i = 0; i < ts_node_named_child_count(params); ++i) {
                if (!check_node_type(ts_node_named_child(params, i), "type")) {
                    SPDLOG_ERROR("Unexpected Error");
                    continue;
                }

                // 解析类型
                TSNode type2 = ts_node_named_child(ts_node_named_child(params, i), 0);
                if (!check_node_type(type2, "subscript")) {
                    SPDLOG_ERROR("Unexpected ERROR");
                    continue;
                }

                std::string typeStr;
                for (uint32_t j = 0; j < ts_node_child_count(type2); ++j) {
                    if (check_node_type(ts_node_child(type2, j), "attribute")) {
                        std::string temp = get_node_text(ts_node_child(type2, j), buffer);
                        typeStr += replace_import_alias(temp, importAlias);
                    } else {
                        typeStr += get_node_text(ts_node_child(type2, j), buffer);
                    }
                }

                returnTypes.emplace_back(typeStr);
            }
        }
        else if (ts_node_named_child_count(type) == 1 && check_node_type(ts_node_named_child(type, 0), "subscript")) {
            // 解析类型
            TSNode type2 = ts_node_named_child(type, 0);

            std::string typeStr;
            for (uint32_t j = 0; j < ts_node_child_count(type2); ++j) {
                if (check_node_type(ts_node_child(type2, j), "attribute")) {
                    std::string temp = get_node_text(ts_node_child(type2, j), buffer);
                    typeStr += replace_import_alias(temp, importAlias);
                } else {
                    typeStr += get_node_text(ts_node_child(type2, j), buffer);
                }
            }

            returnTypes.emplace_back(typeStr);
        }
        else {
            UNIMPLEMENTED_ERROR(type, buffer);
        }

        ret->add_return_type_str(returnTypes);
    }

    // 最后一个child是block
    TSNode block = ts_node_named_child(node, ts_node_named_child_count(node) - 1);
    if (!check_node_type(block, "block")) {
        SPDLOG_ERROR("Unexpected Error");
    }

    ret->add_statement(parse_block_node(block, buffer, importAlias));


    return ret;
}

static pto_parser::PTO_CLASS* create_class_node(TSNode node, const std::string& buffer, const pto_parser::STR_STR_MAP& importAlias) {
    if (!check_node_type(node, "class_definition")) {
        SPDLOG_ERROR("create_class_node can only used for class_definition node");
        return nullptr;
    }

    // 应当只有两个named child
    if (ts_node_named_child_count(node) != 2 || !check_node_type(ts_node_named_child(node, 0), "identifier") || !(check_node_type(ts_node_named_child(node, 1), "block"))) {
        SPDLOG_ERROR("Unexpected Error");
        return nullptr;
    }

    pto_parser::PTO_CLASS* ret = new pto_parser::PTO_CLASS(
        get_node_text(ts_node_named_child(node, 0), buffer),
        ts_node_start_point(node).row + 1,
        ts_node_start_point(node).column
    );

    // 遍历BLOCK节点下的所有child节点
    TSNode block = ts_node_named_child(node, 1);
    for (uint32_t i = 0; i < ts_node_named_child_count(block); ++i) {
        TSNode content = ts_node_named_child(block, i);

        if (check_node_type(content, "decorated_definition")) {
            // 拿到decorator
            std::string decorator = get_decorate_string(ts_node_named_child(content, 0), buffer, importAlias);
            
            TSNode definition = ts_node_named_child(content, 1);
            if (check_node_type(definition, "function_definition")) {
                auto ptr = create_func_node(definition, buffer, importAlias);
                ptr->add_decoration(decorator);
                ret->add_function_def(ptr);
            }
            else {
                UNIMPLEMENTED_ERROR(definition, buffer);
            }
        }
        else {
            UNIMPLEMENTED_ERROR(content, buffer);
        }
    }

    return ret;
}

pto_parser::PTO_MODULE* parse_input_file(const std::string& file, const bool& debug) {
    // 先构建parser
    TSParser *parser = ts_parser_new();
    if (!ts_parser_set_language(parser, tree_sitter_python())) {
        SPDLOG_ERROR("Failed to load Python parser.");
        return nullptr;
    }

    // 将文件内容复制到buffer内
    std::ifstream ifs(file, std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        SPDLOG_ERROR("Failed to open {}", file);
        return nullptr;
    }

    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(size);
    if (!ifs.read(&buffer[0], size)) {
        ifs.close();
        return nullptr;
    }
    ifs.close();

    // 解析文件
    TSTree* tree = ts_parser_parse_string(parser, nullptr, buffer.c_str(), static_cast<uint32_t>(buffer.size()));
    ts_parser_delete(parser);
    if (!tree) {
        SPDLOG_ERROR("Tree-sitter parsing failed.");
        return nullptr;
    }

    if (debug) {
        dump_tree_for_debug(ts_tree_root_node(tree), buffer, 0);
    }

    // 是否有格式错误
    if (ts_node_has_error(ts_tree_root_node(tree))) {
        SPDLOG_ERROR("Syntax error, check debug output.");
        return nullptr;
    }

    // 转换成MODULE_NODE
    pto_parser::PTO_MODULE *ret = new pto_parser::PTO_MODULE();

    TSTreeCursor cursor = ts_tree_cursor_new(ts_tree_root_node(tree));

    // 进入module开始DFS
    ts_tree_cursor_goto_first_child(&cursor);

    // 记录import alias
    pto_parser::STR_STR_MAP importAlias;

    while (true) {
        // 这层循环只处理module的第一层子节点，这些子节点的内部处理需要其他函数
        TSNode node = ts_tree_cursor_current_node(&cursor);

        if (check_node_type(node, "comment")) {
            // 忽略comment
        }
        else if (check_node_type(node, "import_statement")) {
            // 处理import语句, 将可能的别名存在importAlias内
            // 简化处理，这里只会处理别名
            handle_import(node, buffer, importAlias);
        } else if (check_node_type(node, "expression_statement")) {
            // 全局变量
            pto_parser::PTO_ASSIGNMENT *assignment = create_assignment(node, buffer, importAlias);
            ret->add_global_variable(assignment);
            
        } else if (check_node_type(node, "decorated_definition")) {
            // 函数或类定义
            
            // 会有两个节点，一个是decorator，一个是具体的definition
            if (ts_node_named_child_count(node) != 2) {
                SPDLOG_ERROR("Unexpected error");
            } else {
                // 拿到decorator
                std::string decorator = get_decorate_string(ts_node_named_child(node, 0), buffer, importAlias);
   
                TSNode definition = ts_node_named_child(node, 1);
                if (check_node_type(definition, "class_definition")) {
                    auto ptr = create_class_node(definition, buffer, importAlias);
                    ptr->add_decoration(decorator);
                    ret->add_class_or_func(ptr);
                }
                else {
                    UNIMPLEMENTED_ERROR(definition, buffer);
                }

            }
        } else {
            // 未知类型，需补充处理函数
            UNIMPLEMENTED_ERROR(node, buffer);
            ts_tree_cursor_delete(&cursor);
            delete ret;
            return nullptr;
        }
        
        // 是否还有兄弟节点？
        if (!ts_tree_cursor_goto_next_sibling(&cursor))
            break;
    }


    ts_tree_cursor_delete(&cursor);
    ts_tree_delete(tree);
    return ret;
}