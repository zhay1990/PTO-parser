#pragma once
#include <string>
#include <optional>

namespace pto_parser {

struct AppOptions {
    std::string input_file;  // 输入文件路径
    bool debug = false;      // 是否开启调试模式
    std::string output_dir = "./out"; // 默认输出文件夹
};

/**
 * @brief 解析命令行参数并执行环境检查（如创建目录）
 * * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @param options 输出参数，解析成功后的配置将填充至此
 * * @return std::optional<int> 
 * - std::nullopt: 解析成功，请继续执行后续业务逻辑。
 * - int 值: 应当停止执行并返回该退出码（例如 --help 返回 0，参数错误返回 1）。
 */
std::optional<int> parse_arguments(int argc, char** argv, AppOptions& options);

} // namespace pto