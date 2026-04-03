#pragma once

#include "options.hh"
#include <tree_sitter/api.h>
#include <string>

namespace pto_parser {

class PTOAnalyzer {
public:
    explicit PTOAnalyzer(const AppOptions& opts);
    ~PTOAnalyzer();

    // 禁用拷贝
    PTOAnalyzer(const PTOAnalyzer&) = delete;
    PTOAnalyzer& operator=(const PTOAnalyzer&) = delete;

    bool run();

private:
    const AppOptions& options_;
    TSParser* parser_ = nullptr;
    TSTree* tree_ = nullptr;

    std::string read_file(const std::string& path);
    
    // 静态辅助函数：递归打印 AST 结构
    static void print_node_recursive(TSNode node, const std::string& source, int depth);
};

} // namespace pto_parser