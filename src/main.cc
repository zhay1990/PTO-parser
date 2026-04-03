#include "options.hh"
#include "logger.hh"
#include "analyzer.hh"

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

    // 解析输入文件
    pto_parser::PTOAnalyzer analyzer(options);
    analyzer.run();

    return 0;
}