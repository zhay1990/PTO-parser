#include "options.hh"
#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <optional>

namespace pto_parser {

namespace fs = std::filesystem;

std::optional<int> parse_arguments(int argc, char** argv, AppOptions& options) {
    CLI::App app{"PTO-parser: A pyPTO Kernel Static Analyzer"};

    // 1. 输入文件：必填，且检查文件是否存在
    app.add_option("input", options.input_file, "Path to pyPTO file")
       ->required()
       ->check(CLI::ExistingFile);

    // 2. Debug 标志：可选
    app.add_flag("-d,--debug", options.debug, "Enable debug logging and symbols");

    // 3. 输出文件夹 (选填，带默认值)
    app.add_option("-o,--output-dir", options.output_dir, "Directory for all output files")
       ->capture_default_str();

    try {
        app.parse(argc, argv);
        
        // --- 核心逻辑：在解析成功后自动创建文件夹 ---
        fs::path out_path(options.output_dir);
        if (!fs::exists(out_path)) {
            // 使用 create_directories 递归创建
            if (fs::create_directories(out_path)) {
                // 解析阶段通常还没初始化 spdlog，所以用 std::cout 或保持静默
            }
        }

    } catch (const CLI::ParseError &e) {
        // app.exit(e) 对于 --help 返回 0, 对于错误返回 1 或其他
        return app.exit(e);
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error: Failed to create output directory: " << e.what() << std::endl;
        return 1;
    }

    return std::nullopt;
}

} // namespace pto