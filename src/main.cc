#include "options.hh"
#include "logger.hh"
#include "parser.hh"
#include <filesystem>


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

    // 处理变量的类型，并做正确性检查
    if (!module->type_check()) {
        delete module;
        return 1;
    }

    // 死代码消除
    if (!module->dead_code_eliminate()) {
        delete module;
        return 1;
    }
    
    // 将优化过的代码输出到文件
    std::filesystem::path ptoFile = std::filesystem::path(options.output_dir) /
                                    (std::filesystem::path(options.input_file).stem().string() + "_opt.py");
    
    std::ofstream fout(ptoFile, std::ios::out);

    if (!fout.is_open()) {
        SPDLOG_ERROR("Failed to open file: {}", ptoFile.string());
        delete module;
        return 1;
    }

    module->dump(0, fout);

    fout.close();

    // Clean up
    delete module;
    SPDLOG_INFO("Clean up succeffully. Exited.");
    return 0;
}