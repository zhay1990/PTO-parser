#pragma once
#include <string>
#include <spdlog/spdlog.h>

namespace pto_parser {

/**
 * @brief 初始化全局 logger
 * @param log_file_path 日志文件保存路径
 * @param debug_mode 如果为 true，文件日志将记录 DEBUG 级别，否则为 INFO
 */
void setup_logger(const std::string& log_file_path, bool debug_mode);

} // namespace pto