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

    SPDLOG_INFO("Static type check completed!");

    // 在进行死代码消除前，将for循环中的初始变量和yield语句拆成单独的assignment语句
    // 这是因为yield语句会导致函数调用时的循环依赖，即
    // for xxx, (A, B) in pl.range(20, init_values(xxx, xxx)):
    //     ret = self.xxx(A, B)
    //     C = ret[0]
    //     D = ret[1]
    //     E, F = pl.yield_(C, D)
    // 然后self.xxx中A只用来生成C
    // 在这种情况下，即使后续没有使用E，我们也无法消除E
    // 因为A是函数需要的，导致E一定要存在
    // 为了处理这种情况，我们将yield和init value拆成多个assignment语句
    if (!module->remove_yield()) {
        delete module;
        return 1;
    }

    SPDLOG_INFO("Yield statements are replaced by assignment.");

    // 死代码消除
    if (!module->dead_code_eliminate()) {
        delete module;
        return 1;
    }

    SPDLOG_INFO("Dead code eliminate completed");

    // 做一次静态检查
    if (!module->type_check()) {
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