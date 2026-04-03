#include "analyzer.hh"
#include "logger.hh"
#include <fstream>
#include <sstream>
#include <algorithm>

extern "C" const TSLanguage* tree_sitter_python();

namespace pto_parser {

PTOAnalyzer::PTOAnalyzer(const AppOptions& opts) : options_(opts) {
    parser_ = ts_parser_new();
    if (!ts_parser_set_language(parser_, tree_sitter_python())) {
        SPDLOG_ERROR("pto_parser: Failed to load Python grammar.");
        throw std::runtime_error("Language load failed");
    }
    SPDLOG_DEBUG("pto_parser: PTOAnalyzer core initialized.");
}

PTOAnalyzer::~PTOAnalyzer() {
    if (tree_) ts_tree_delete(tree_);
    if (parser_) ts_parser_delete(parser_);
    SPDLOG_DEBUG("pto_parser: PTOAnalyzer resources released.");
}

void PTOAnalyzer::print_node_recursive(TSNode node, const std::string& source, int depth) {
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
    
    // 打印到日志文件（带文件行号）和终端（简洁版）
    SPDLOG_INFO("{}[{}] L{}: \"{}\"", indent, type, point.row + 1, text);

    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        print_node_recursive(ts_node_child(node, i), source, depth + 1);
    }
}

bool PTOAnalyzer::run() {
    std::string source = read_file(options_.input_file);
    if (source.empty()) return false;

    SPDLOG_INFO("pto_parser: Parsing file {}", options_.input_file);

    tree_ = ts_parser_parse_string(parser_, nullptr, source.c_str(), static_cast<uint32_t>(source.size()));
    if (!tree_) {
        SPDLOG_ERROR("pto_parser: Tree-sitter parsing failed.");
        return false;
    }

    TSNode root = ts_tree_root_node(tree_);
    
    // 打印 AST 结构以供确认
    SPDLOG_INFO("--- [AST Structure Dump] ---");
    print_node_recursive(root, source, 0);
    SPDLOG_INFO("--- [Dump End] ---");

    return true;
}

std::string PTOAnalyzer::read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        SPDLOG_ERROR("pto_parser: Failed to open {}", path);
        return "";
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

} // namespace pto_parser