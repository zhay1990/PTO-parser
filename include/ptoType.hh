#pragma once
#include <string>
#include <vector>

namespace pto_parser {

enum class PTO_TYPE_KIND {
    UNKNOWN,
    VOID_STMT,  // 语句类型（无返回值，如 if, for, assign）
    CONST_INT,
    CONST_FLOAT,
    CONST_BOOL,
    TENSOR,     // 张量
    TUPLE,      // 元组 (在我们的实现中，() 和 [] 统一映射为 TUPLE)
    BF16,
    FP32,
    INT32,
    BOOL
};

class PTO_TYPE {
public:
    PTO_TYPE_KIND kind;
    std::vector<int> shape;          // 仅 TENSOR 使用，如果是-1则表示是动态shape
    std::vector<std::string> dynamicShape; // 仅TENSOR使用，按顺序表示某一-1维度的变量名
    std::vector<PTO_TYPE> sub_types; // TENSOR (存数据类型) 或 TUPLE (存内部元素类型)
    bool isDynamic; // 用于声明该变量可以用于定义tensor的动态shape

    PTO_TYPE(PTO_TYPE_KIND k = PTO_TYPE_KIND::UNKNOWN);

    // 静态工厂函数
    static PTO_TYPE make_scalar(PTO_TYPE_KIND k);
    static PTO_TYPE make_tensor(const std::vector<int>& shape, const PTO_TYPE& elem_type);
    static PTO_TYPE make_tuple(const std::vector<PTO_TYPE>& elements);

    // 运算符重载用于类型比较
    bool operator==(const PTO_TYPE& other) const;
    bool operator!=(const PTO_TYPE& other) const;
    
    std::string to_string() const;
};

} // namespace pto_parser