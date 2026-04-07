#include "parser.hh"
#include "logger.hh"
#include "ast_process.hh"
#include <fstream>
#include <tree_sitter/api.h>
#include <string.h>

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

static inline const std::string get_attribute_str(TSNode node, const std::string& buffer, const std::unordered_map<std::string, std::string>& importMap) {
    if (ts_node_is_null(node)) return "";

    if (!check_node_type(node, "attribute")) {
        SPDLOG_ERROR("get_attribute_str only used for attribute node, but got {} node", ts_node_type(node));
        return "";
    }

    std::string ret;

    if (check_node_type(ts_node_named_child(node, 0), "attribute")) {
        ret = get_attribute_str(ts_node_named_child(node, 0), buffer, importMap);
    }
    else if (check_node_type(ts_node_named_child(node, 0), "identifier")) {
        ret = get_node_text(ts_node_named_child(node, 0), buffer);
        if (importMap.find(ret) != importMap.end()) {
           ret = importMap.find(ret)->second;
        }
    }
    else {
        SPDLOG_ERROR("Unexpected node type {}", ts_node_type(node));
        return "";
    }

    for (uint32_t i = 1; i < ts_node_named_child_count(node); i ++) {
        ret += "." + get_node_text(ts_node_named_child(node, i), buffer);
    }

    return ret;
}

extern "C" const TSLanguage* tree_sitter_python();

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

static void gen_func_call(struct FUNCTION_CALL& callNode, TSNode node, const std::string& buffer, const std::unordered_map<std::string, std::string>& importMap) {
    if (!check_node_type(node, "call")) {
        SPDLOG_ERROR("gen_func_call only process call node");
        return;
    }

    // 第一个节点应当是identifier或attribute
    if (check_node_type(ts_node_named_child(node, 0), "identifier")) {
        callNode.funcName = get_node_text(ts_node_named_child(node, 0), buffer);
    } 
    else if (check_node_type(ts_node_named_child(node, 0), "attribute")) {
        // 函数调用名在attribute中，第一个identifier可能需要用import table调整
        callNode.funcName = get_attribute_str(ts_node_named_child(node, 0), buffer, importMap);
    }
    else {
        SPDLOG_ERROR("Process method for '{}' at line {} is not implemented",
            get_node_text(node, buffer), ts_node_start_point(node).row + 1
        );
    }

    // 处理argument list
    TSNode argumentList = ts_node_named_child(node, 1);
    for (uint32_t i = 0; i < ts_node_named_child_count(argumentList); i ++) {
        // 暂时只处理string类型
        TSNode param = ts_node_named_child(argumentList, i);

        if (check_node_type(param, "string")) {
            struct CALL_ARGUMENT argument;
            argument.type = CALL_ARGUMENT_TYPE::VARIABLE;
            argument.varName = get_node_text(param, buffer);
            callNode.arguments.emplace_back(argument);
        }
        else if (check_node_type(param, "keyword_argument")) {
            struct CALL_ARGUMENT argument;
            argument.type = CALL_ARGUMENT_TYPE::KEYWORD;
            
            TSNode var = ts_node_named_child(param, 0);
            TSNode val = ts_node_named_child(param, 1);

            if (check_node_type(var, "identifier")) {
                argument.varName = get_node_text(var, buffer);
            }
            else if (check_node_type(var, "attribute")) {
                argument.varName = get_attribute_str(var, buffer, importMap);
            } else {
                SPDLOG_ERROR("Process method for '{}' at line {} is not implemented",
                    get_node_text(node, buffer), ts_node_start_point(node).row + 1
                );
            }

            if (check_node_type(val, "identifier")) {
                argument.value = get_node_text(val, buffer);
            }
            else if (check_node_type(val, "attribute")) {
                argument.value = get_attribute_str(val, buffer, importMap);
            } else {
                SPDLOG_ERROR("Process method for '{}' at line {} is not implemented",
                    get_node_text(node, buffer), ts_node_start_point(node).row + 1
                );
            }

            callNode.arguments.emplace_back(argument);
        }
        else {
            SPDLOG_ERROR("Process method for function call '{}' at line {} is not implemented.",
                get_node_text(node, buffer),
                ts_node_start_point(node).row + 1  
            );
            return;
        }
    }
}

static const std::string get_decorate_string(TSNode node, const std::string& buffer, const std::unordered_map<std::string, std::string>& importMap) {
    if (!check_node_type(node, "decorator")) {
        SPDLOG_ERROR("get_decorate_string only process decorate node");
        return "";
    }

    std::string decorate = "";

    // 现在看到了三种情况
    if (ts_node_named_child_count(node) == 1) {
        if (strcmp(ts_node_type(ts_node_named_child(node, 0)), "attribute") == 0) {
            TSNode attribute = ts_node_named_child(node, 0);
            // xx.xx形式，可能需要替换alias
            decorate = get_node_text(ts_node_named_child(attribute, 0), buffer);
            
            if (importMap.find(decorate) != importMap.end()) {
                decorate = importMap.find(decorate)->second;
            }

            for (uint32_t i = 1; i < ts_node_named_child_count(attribute); i ++) {
                decorate += "." + get_node_text(ts_node_named_child(attribute, i), buffer);
            }
        }
        else if (strcmp(ts_node_type(ts_node_named_child(node, 0)), "identifier") == 0) {
            decorate = get_node_text(ts_node_named_child(node, 0), buffer);
        }
        else if (strcmp(ts_node_type(ts_node_named_child(node, 0)), "call") == 0) {
            struct FUNCTION_CALL tempNode;
            gen_func_call(tempNode, ts_node_named_child(node, 0), buffer, importMap);
            
            // 转成string返回
            decorate = tempNode.funcName + "(";

            for (std::size_t i = 0; i < tempNode.arguments.size(); i ++) {
                decorate += tempNode.arguments[i].varName;

                switch (tempNode.arguments[i].type) {
                    case CALL_ARGUMENT_TYPE::VARIABLE: break;
                    case CALL_ARGUMENT_TYPE::KEYWORD: decorate += "=" + tempNode.arguments[i].value; break;
                    default: SPDLOG_ERROR("Error!");
                }
                decorate += ",";
            }
            decorate.back() = ')';
        }
        else {
            SPDLOG_ERROR("Process method for '{}' at line {} is not implemented",
                get_node_text(node, buffer), ts_node_start_point(node).row + 1
            );
        }
    }

    return decorate;
}
                

static bool process_aliased_import(TSNode node, std::unordered_map<std::string, std::string>& importMap, const std::string& buffer) {
    // 应当只有一个匿名节点，即aliased_import
    if (ts_node_is_null(node)) {
        SPDLOG_ERROR("Unexpected nullptr");
        return false;
    }

    if (ts_node_named_child_count(node) != 1) return false;

    TSNode child = ts_node_named_child(node, 0);
    if (strcmp(ts_node_type(child), "aliased_import") != 0) return false;

    // 这个节点有两个命名节点
    if (ts_node_named_child_count(child) != 2) {
        SPDLOG_ERROR("Unexpected error for aliased import");
        // 算处理过了
        return true;
    }

    std::string aliasName = get_node_text(ts_node_named_child(child, 1), buffer);
    std::string originalName = get_node_text(ts_node_named_child(child, 0), buffer);

    SPDLOG_DEBUG("Got alias import name '{}' for '{}'", aliasName, originalName);

    if (importMap.find(aliasName) != importMap.end()) {
        SPDLOG_ERROR("Duplicated alias name '{}' for '{}' and '{}' at line {}",
            aliasName, originalName, importMap[aliasName],
            ts_node_start_point(node).row + 1
        );
        // 如果有重名的话，用最新的覆盖旧的
    }

    importMap[aliasName] = originalName;

    return true;
}

static void handle_import(TSNode node, std::unordered_map<std::string, std::string>& importMap, const std::string& buffer) {
    // 分情况处理
    // 1. import xxx.xx as xx
    // 这种模式下 node有两个子节点，一个import 一个aliased_import
    if (process_aliased_import(node, importMap, buffer)) return;

    // 其他模式待补充
    SPDLOG_ERROR("Process method for statement '{}' at line {} is not implemented.", get_node_text(node, buffer), ts_node_start_point(node).row + 1);
}

static void add_global_variable(TSNode node, std::unordered_map<std::string, struct VARIABLE_NODE*>& globalVariable, const std::string& buffer, const std::unordered_map<std::string, std::string>& globalImportMap) {
    // 应当有一个assignment节点
    if (ts_node_is_null(node)) {
        SPDLOG_ERROR("Unpexected nullptr");
        return;
    }
    if (ts_node_named_child_count(node) == 1 && strcmp(ts_node_type(ts_node_named_child(node, 0)), "assignment") == 0) {
        TSNode assign = ts_node_named_child(node, 0);

        // 处理有两个name节点，且第二个节点是call的statement
        if (ts_node_named_child_count(assign) == 2 && strcmp(ts_node_type(ts_node_named_child(assign, 1)), "call") == 0) {
            // 创建一个全局变量
            struct VARIABLE_NODE *globalVar = new struct VARIABLE_NODE();
            
            std::string varName = get_node_text(ts_node_named_child(assign, 0), buffer);

            globalVar->name = varName;
            // 函数调用的默认类型为INT，之后再检查函数返回类型
            globalVar->type = VARIABLE_TYPE::INT32;
            
            TSNode call = ts_node_named_child(assign, 1);
            
            gen_func_call(globalVar->call, call, buffer, globalImportMap);

            if (globalVariable.find(varName) != globalVariable.end()) {
                SPDLOG_ERROR("Duplicated variable definition of '{}' at line '{}'",
                    varName, ts_node_start_point(node).row + 1
                );
                delete globalVar;
            } else {
                globalVariable[varName] = globalVar;
            }

            return;
        }
    }

    SPDLOG_ERROR("Process method for statement '{}' at line {} is not implemented.", get_node_text(node, buffer), ts_node_start_point(node).row + 1);

}

struct MODULE_NODE* parse_input_file(const std::string& file, const bool& debug) {
    // 先构建parser
    TSParser *parser = ts_parser_new();
    if (!ts_parser_set_language(parser, tree_sitter_python())) {
        SPDLOG_ERROR("Failed to load Python parser.");
        return nullptr;
    }

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

    TSTree* tree = ts_parser_parse_string(parser, nullptr, buffer.c_str(), static_cast<uint32_t>(buffer.size()));
    ts_parser_delete(parser);
    if (!tree) {
        SPDLOG_ERROR("Tree-sitter parsing failed.");
        return nullptr;
    }

    if (debug) {
        dump_tree_for_debug(ts_tree_root_node(tree), buffer, 0);
    }

    // 转换成MODULE_NODE
    struct MODULE_NODE *ret = nullptr;

    TSTreeCursor cursor = ts_tree_cursor_new(ts_tree_root_node(tree));

    // 这里假定了一个python文件只有一个module
    if (strcmp(ts_node_type(ts_tree_cursor_current_node(&cursor)), "module") != 0) {
        SPDLOG_ERROR("The type of root TSnode should be 'module', but got '{}'", ts_node_type(ts_tree_cursor_current_node(&cursor)));
        return ret;
    }

    // 进入module开始DFS
    ts_tree_cursor_goto_first_child(&cursor);

    // 记录文件中import的别名
    std::unordered_map<std::string, std::string> globalImportMap;
    std::unordered_map<std::string, struct VARIABLE_NODE*> globalVariable;

    while (true) {
        // 这层循环只处理module的第一层子节点，这些子节点的内部处理需要其他函数
        TSNode node = ts_tree_cursor_current_node(&cursor);
        const char* type = ts_node_type(node);

        if (strcmp(type, "comment") == 0) {
            // 忽略comment
        }
        else if (strcmp(type, "import_statement") == 0) {
            // 处理import语句
            handle_import(node, globalImportMap, buffer);
        } else if (strcmp(type, "expression_statement") == 0) {
            // 全局变量
            add_global_variable(node, globalVariable, buffer, globalImportMap);
        } else if (strcmp(type, "decorated_definition") == 0) {
            // 函数或类定义
            
            // 会有两个节点，一个是decorator，一个是具体的definition
            if (ts_node_named_child_count(node) != 2) {
                SPDLOG_ERROR("Unexpected error");
            } else {
                // 拿到decorator
                std::string decorator = get_decorate_string(ts_node_named_child(node, 0), buffer, globalImportMap);
                
                TSNode definition = ts_node_named_child(node, 1);
                if (strcmp(ts_node_type(definition), "class_definition") == 0) {

                }
                else {
                    SPDLOG_ERROR("Process method for {} node is not implemented", ts_node_type(definition));
                }

            }
        } else {
            // 未知类型，需补充处理函数
            SPDLOG_ERROR("Unexpected node type '{}'", type);
            ts_tree_cursor_delete(&cursor);
            ast_module_delete(ret);
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

void FUNCTION_CALL::dump() {
    SPDLOG_INFO("FuncName = {}", funcName);
    for (std::size_t i = 0; i < arguments.size(); i ++) {
        switch (arguments[i].type) {
            case CALL_ARGUMENT_TYPE::VARIABLE:
                SPDLOG_INFO("Argument #{}: {}", i, arguments[i].varName);
                break;
            case CALL_ARGUMENT_TYPE::KEYWORD:
                SPDLOG_INFO("Argument #{}: {} = {}", i, arguments[i].varName, arguments[i].value);
                break;
        }
    }
}