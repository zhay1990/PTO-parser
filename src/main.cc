#include "options.hh"
#include "logger.hh"
#include <iostream>
#include <string>
#include <vector>
#include <tree_sitter/api.h>

// 必须包裹在 extern "C" 中，因为这些函数是 C 语言实现的
extern "C" {
    TSLanguage *tree_sitter_python();
}

int main(int argc, char** argv) {
    pto_parser::AppOptions options;

    // 执行参数解析
    auto exit_code = pto_parser::parse_arguments(argc, argv, options);
    if (exit_code.has_value()) {
        return exit_code.value();
    }

    pto_parser::setup_logger(options.output_dir, options.debug);

    
    SPDLOG_INFO("PTO-parser started successfully.");
    SPDLOG_INFO("Input file: {}", options.input_file);
    SPDLOG_INFO("Output directory: {}", options.output_dir);

    if (options.debug) {
        SPDLOG_DEBUG("Debug mode is ON. Detailed AST info will be logged to file.");
    }


    return 0;
}