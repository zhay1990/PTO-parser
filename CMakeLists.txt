cmake_minimum_required(VERSION 3.10)
project(PTO-parser LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

# 1. 引入子工程
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog)
add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/CLI11)

# 2. 包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter/lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter/lib/src
)

# 3. 自动化收集 src 目录下的所有 C/C++ 文件
file(GLOB_RECURSE PROJECT_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

# 4. 补充三方库中必须手动指定的非 src 文件 (如 tree-sitter 核心)
set(ALL_SOURCES
    ${PROJECT_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter/lib/src/lib.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter-python/src/parser.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter-python/src/scanner.c
)

add_executable(pto_parser ${ALL_SOURCES})

# 5. 链接所有三方库
# CLI11::CLI11 是其官方导出的 target
target_link_libraries(pto_parser PRIVATE 
    spdlog::spdlog
    CLI11::CLI11
)

# 6. 设置默认构建类型（如果用户没指定，默认为 Release）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 7. 针对不同模式配置编译选项
# $<CONFIG:Debug> 是生成器表达式，仅在 Debug 模式下生效
target_compile_options(pto_parser PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG_MODE>
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    -Wall -Wextra
)

# 8. 打印当前构建类型，方便确认
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter-python/src/parser.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tree-sitter-python/src/scanner.c 
    PROPERTIES COMPILE_FLAGS "-w"
)



# 9. 在构建完成后自动拷贝二进制文件到根目录
add_custom_command(TARGET pto_parser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:pto_parser>
            ${CMAKE_CURRENT_SOURCE_DIR}/pto_parser
    COMMENT "Copying pto_parser to project root..."
)

# 10. 配置清理时也删除根目录的文件
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/pto_parser")