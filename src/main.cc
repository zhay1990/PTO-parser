#include "options.hh"
#include "logger.hh"
#include "parser.hh"


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

    // 解析输入的python文件，转化成MODULE结构
    pto_parser::PTO_MODULE *module = parse_input_file(options.input_file, options.debug);
    if (module == nullptr) {
        return 1;
    }

    // Clean up
    delete module;
    SPDLOG_INFO("Clean up succeffully. Exited.");
    return 0;
}