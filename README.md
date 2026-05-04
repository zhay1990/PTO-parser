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
Markdown
## 环境依赖：LLVM/MLIR 底座独立构建指南

本工程依赖 LLVM/MLIR 框架（`release/22.x`）。由于 LLVM 体积庞大，请**不要**将其作为子模块引入工程，而是按照以下步骤在本地独立编译一次。此配置已针对 64GB 内存工作站进行极速编译优化。

### 1. 系统依赖准备 (Ubuntu / Debian)
为了启用极速链接和编译，请确保安装了 Clang 工具链和 LLD 链接器：
```bash
sudo apt update
sudo apt install clang lld ninja-build cmake git
2. 获取源码 (浅克隆)
为了节省网络流量和约 3GB 的磁盘空间，请使用浅克隆仅拉取最新快照：

Bash
cd ~  # 推荐在独立的工作区执行，不要放在 pto_compiler 目录内
git clone --depth 1 --branch release/22.x [https://github.com/llvm/llvm-project.git](https://github.com/llvm/llvm-project.git)
3. CMake 满血版配置
进入目录并创建 build 文件夹进行树外构建。此配置仅编译本机 CPU 与 NVPTX 后端，并启用了 LLD 多核链接。

Bash
cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON
4. 全核极速编译
在 64GB 内存环境下，直接执行 ninja 全核心并发编译（耗时约 10-15 分钟）：

Bash
ninja
⚠️ 核心注意事项
禁止 Install：编译完成后，绝对不要执行 sudo ninja install。我们将直接引用 build 目录下的产物。

构建链接：在编译 pto_compiler 时，CMake 会自动通过 -DLLVM_DIR 和 -DMLIR_DIR 找到此处的库文件。请确保不要随意移动或删除此 llvm-project 文件夹。

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