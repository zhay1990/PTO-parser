# PTO-parser

**PTO-parser** 是一个基于 [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) 的高性能 pyPTO 静态分析工具。

-----

## 🚀 快速开始

### 1\. 克隆项目 (包含子模块)

本项目使用了 Git 子模块来管理第三方依赖（Tree-sitter, spdlog, CLI11），克隆时**必须**带上 `--recursive` 参数：

```bash
git clone --recursive git@github.com:你的用户名/PTO-parser.git
cd PTO-parser
```

> **补救措施**：如果你克隆时忘记带参数，导致 `third_party` 文件夹为空，请执行：
>
> ```bash
> git submodule update --init --recursive
> ```

### 2\. 编译与构建

推荐使用标准 CMake 流程进行构建。

#### 编译 Release 版本 (生产/分析用)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

#### 编译 Debug 版本 (GDB 调试用)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug ..
cmake --build build -j$(nproc)
```

### 3\. 清理构建

执行以下命令将同时清理 `build/` 目录下的中间文件以及根目录下的二进制副本：

```bash
cd build
make clean
```

-----

## 📖 使用说明

构建完成后，二进制文件会自动出现在项目根目录下。

```bash
# 查看帮助文档
./pto_parser --help

```

-----

## 📂 项目结构

```text
.
├── src/                # 业务逻辑
├── third_party/        # 第三方依赖
│   ├── tree-sitter/
│   ├── spdlog/
│   └── CLI11/
├── CMakeLists.txt      # 自动化构建脚本
├── pto_parser          # 自动生成的 Release 二进制文件
└── README.md
```

-----

## 🤝 开发指南

  * **添加新功能**: 直接在 `src/` 目录下创建新文件即可，无需修改 `CMakeLists.txt`。
  * **调试**: 在 `Debug` 模式下，可以使用 `gdb ./pto_parser` 进行断点调试。
  * **警告处理**: 自动生成的 `parser.c` 等三方代码已在构建中被设置为“静默”模式（`-w`），你只需关注 `src/` 逻辑中的警告信息。

-----

## 📄 许可

[MIT License](https://www.google.com/search?q=LICENSE)