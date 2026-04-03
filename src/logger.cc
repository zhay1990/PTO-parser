#include "logger.hh"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>

namespace pto_parser {

void setup_logger(const std::string& output_dir, bool debug_mode) {
    namespace fs = std::filesystem;
    try {
        // 直接拼接路径，假设目录已由 parse_arguments 创建
        fs::path log_file = fs::path(output_dir) / "pto_parser.log";

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("[%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file.string(), true);
        file_sink->set_level(debug_mode ? spdlog::level::debug : spdlog::level::info);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%l] %v");

        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("pto_logger", sinks.begin(), sinks.end());
        logger->set_level(debug_mode ? spdlog::level::debug : spdlog::level::info);

        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::err);

    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Logger initialization failed: %s\n", ex.what());
    }
}

} // namespace pto