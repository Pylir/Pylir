
#pragma once

#include <mlir/IR/BuiltinAttributes.h>

#include "PylirTypes.hpp"

namespace pylir::Dialect
{
namespace detail
{
template <class T>
struct ValueAttrStorage : public mlir::AttributeStorage
{
    T value;

    ValueAttrStorage(T value, mlir::Type type) : mlir::AttributeStorage(type), value(value) {}

    using KeyTy = std::pair<T, mlir::Type>;

    bool operator==(KeyTy key) const
    {
        if constexpr (std::is_same_v<llvm::APInt, T>)
        {
            auto maxSize = std::max(key.first.getBitWidth(), value.getBitWidth());
            return key.first.sextOrSelf(maxSize) == value.sextOrSelf(maxSize);
        }
        else
        {
            return key.first == value && key.second == getType();
        }
    }

    static llvm::hash_code hashKey(KeyTy key)
    {
        return llvm::hash_combine(key.first, key.second);
    }

    static ValueAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        return new (allocator.allocate<ValueAttrStorage>()) ValueAttrStorage(value.first, value.second);
    }
};

struct TypeStorage : public mlir::AttributeStorage
{
    TypeStorage(mlir::Type type) : mlir::AttributeStorage(type) {}

    using KeyTy = mlir::Type;

    bool operator==(KeyTy key) const
    {
        return getType() == key;
    }

    static llvm::hash_code hashKey(KeyTy key)
    {
        return mlir::hash_value(key);
    }

    static TypeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        return new (allocator.allocate<TypeStorage>()) TypeStorage(value);
    }
};

} // namespace detail

class NoneAttr : public mlir::Attribute::AttrBase<NoneAttr, mlir::Attribute, detail::TypeStorage>
{
public:
    using Base::Base;

    static NoneAttr get(mlir::MLIRContext* context)
    {
        return Base::get(context, NoneType::get(context));
    }
};

class BoolAttr : public mlir::Attribute::AttrBase<BoolAttr, mlir::Attribute, detail::ValueAttrStorage<bool>>
{
public:
    using Base::Base;

    static BoolAttr get(mlir::MLIRContext* context, bool value)
    {
        return Base::get(context, value, BoolType::get(context));
    }

    bool getValue()
    {
        return getImpl()->value;
    }
};

class FloatAttr : public mlir::Attribute::AttrBase<FloatAttr, mlir::Attribute, detail::ValueAttrStorage<llvm::APFloat>>
{
public:
    using Base::Base;

    static FloatAttr get(mlir::MLIRContext* context, double value)
    {
        return Base::get(context, value, FloatType::get(context));
    }

    double getValue()
    {
        return getImpl()->value.convertToDouble();
    }
};

class IntegerAttr
    : public mlir::Attribute::AttrBase<IntegerAttr, mlir::Attribute, detail::ValueAttrStorage<llvm::APInt>>
{
public:
    using Base::Base;

    static IntegerAttr get(mlir::MLIRContext* context, const llvm::APInt& value)
    {
        return Base::get(context, value, IntegerType::get(context));
    }

    const llvm::APInt& getValue()
    {
        return getImpl()->value;
    }
};

class StringAttr : public mlir::Attribute::AttrBase<StringAttr, mlir::Attribute, detail::ValueAttrStorage<std::string>>
{
public:
    using Base::Base;

    static StringAttr get(mlir::MLIRContext* context, std::string value)
    {
        return Base::get(context, std::move(value), StringType::get(context));
    }

    llvm::StringRef getValue()
    {
        return getImpl()->value;
    }
};

} // namespace pylir::Dialect
