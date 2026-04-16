#include "ptoType.hh"

namespace pto_parser {

PTO_TYPE::PTO_TYPE(PTO_TYPE_KIND k)
    : kind(k),
      shape(),
      dynamicShape(),
      sub_types(),
      isDynamic(false)
{}

PTO_TYPE PTO_TYPE::make_scalar(PTO_TYPE_KIND k) {
    return PTO_TYPE(k);
}

PTO_TYPE PTO_TYPE::make_tensor(const std::vector<int>& shape, const PTO_TYPE& elem_type) {
    PTO_TYPE t(PTO_TYPE_KIND::TENSOR);
    t.shape = shape;
    t.sub_types.push_back(elem_type);
    t.isDynamic = false;
    return t;
}

PTO_TYPE PTO_TYPE::make_tuple(const std::vector<PTO_TYPE>& elements) {
    PTO_TYPE t(PTO_TYPE_KIND::TUPLE);
    t.sub_types = elements;
    t.isDynamic = false;
    return t;
}

bool PTO_TYPE::operator==(const PTO_TYPE& other) const {
    if (kind != other.kind) return false;

    // 复合类型检查形状
    if (shape != other.shape) return false;
    if (dynamicShape != other.dynamicShape) return false;
    
    // 递归检查子类型（TENSOR 的 ElementType，或 TUPLE 的内部元素）
    if (sub_types.size() != other.sub_types.size()) return false;
    for (size_t i = 0; i < sub_types.size(); ++i) {
        if (sub_types[i] != other.sub_types[i]) return false;
    }

    // dynamic是个特殊标记，不在这做检查

    return true;
}

bool PTO_TYPE::operator!=(const PTO_TYPE& other) const {
    return !(*this == other);
}

std::string PTO_TYPE::to_string() const {
    switch (kind) {
        case PTO_TYPE_KIND::UNKNOWN: return "unknown";
        case PTO_TYPE_KIND::VOID_STMT: return "void";
        case PTO_TYPE_KIND::CONST_INT: return "const int";
        case PTO_TYPE_KIND::CONST_FLOAT: return "const float";
        case PTO_TYPE_KIND::CONST_BOOL: return "const bool";
        case PTO_TYPE_KIND::TENSOR: {
            std::string res = "tensor<[";
            for (size_t i = 0; i < shape.size(); ++i) {
                res += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");
            }
            res += "], ";
            if (dynamicShape.size() != 0) {
                res += "[";
                for (size_t i = 0; i < dynamicShape.size(); ++i) {
                    res += dynamicShape[i] + (i == dynamicShape.size() - 1 ? "" : ", ");
                }
                res += "], ";
            }
            res += (sub_types.empty() ? "unknown" : sub_types[0].to_string()) + ">";
            return res;
        }
        case PTO_TYPE_KIND::TUPLE: {
            std::string res = "tuple[";
            for (size_t i = 0; i < sub_types.size(); ++i) {
                res += sub_types[i].to_string() + (i == sub_types.size() - 1 ? "" : ", ");
            }
            res += "]";
            return res;
        }
        case PTO_TYPE_KIND::BF16: return "BF16";
        case PTO_TYPE_KIND::FP32: return "FP32";
        case PTO_TYPE_KIND::INT32: return "INT32";
        default: return "undefined";
    }
}

} // namespace pto_parser