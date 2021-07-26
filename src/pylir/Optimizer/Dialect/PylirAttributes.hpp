
#pragma once

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

#include "PylirTypes.hpp"

namespace pylir::Dialect
{
namespace detail
{
struct IntegerStorage : public mlir::AttributeStorage
{
    llvm::APInt value;

    IntegerStorage(llvm::APInt value) : value(value) {}

    using KeyTy = llvm::APInt;

    bool operator==(KeyTy key) const
    {
        return key == value;
    }

    static IntegerStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        return new (allocator.allocate<IntegerStorage>()) IntegerStorage(value);
    }
};

struct ListStorage : public mlir::AttributeStorage
{
    llvm::ArrayRef<mlir::Attribute> value;

    ListStorage(llvm::ArrayRef<mlir::Attribute> value, mlir::Type type) : mlir::AttributeStorage(type), value(value) {}

    using KeyTy = std::pair<llvm::ArrayRef<mlir::Attribute>, mlir::Type>;

    bool operator==(KeyTy key) const
    {
        return key.first == value && key.second == getType();
    }

    static llvm::hash_code hashKey(KeyTy key)
    {
        return llvm::hash_value(key);
    }

    static ListStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        value.first = allocator.copyInto(value.first);
        return new (allocator.allocate<ListStorage>()) ListStorage(value.first, value.second);
    }
};

struct DictStorage : public mlir::AttributeStorage
{
    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value;

    DictStorage(llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value, mlir::Type type)
        : mlir::AttributeStorage(type), value(value)
    {
    }

    using KeyTy = std::pair<llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>>, mlir::Type>;

    bool operator==(KeyTy key) const
    {
        return key.first == value && key.second == getType();
    }

    static llvm::hash_code hashKey(KeyTy key)
    {
        return llvm::hash_value(key);
    }

    static DictStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        value.first = allocator.copyInto(value.first);
        return new (allocator.allocate<DictStorage>()) DictStorage(value.first, value.second);
    }
};

} // namespace detail

class IntegerAttr : public mlir::Attribute::AttrBase<IntegerAttr, mlir::Attribute, detail::IntegerStorage>
{
public:
    using Base::Base;

    static IntegerAttr get(mlir::MLIRContext* context, const llvm::APInt& value)
    {
        return Base::get(context, value);
    }

    const llvm::APInt& getValue()
    {
        return getImpl()->value;
    }
};

class SetAttr : public mlir::Attribute::AttrBase<SetAttr, mlir::Attribute, detail::ListStorage>
{
public:
    using Base::Base;

    static SetAttr get(mlir::MLIRContext* context, llvm::DenseSet<mlir::Attribute> value)
    {
        return Base::get(context, std::vector<mlir::Attribute>{value.begin(), value.end()}, SetType::get(context));
    }

    llvm::ArrayRef<mlir::Attribute> getValue()
    {
        return getImpl()->value;
    }
};

class DictAttr : public mlir::Attribute::AttrBase<DictAttr, mlir::Attribute, detail::DictStorage>
{
public:
    using Base::Base;

    static DictAttr get(mlir::MLIRContext* context, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
    {
        llvm::DenseMap<mlir::Attribute, mlir::Attribute> map;
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> values;
        for (auto& iter : value)
        {
            if (map.insert(iter).second)
            {
                values.push_back(iter);
            }
        }
        return Base::get(context, values, DictType::get(context));
    }

    static DictAttr getAlreadySorted(mlir::MLIRContext* context,
                                     llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
    {
        return Base::get(context, value, DictType::get(context));
    }

    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> getValue()
    {
        return getImpl()->value;
    }
};

} // namespace pylir::Dialect
