
#pragma once

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

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
        return llvm::hash_value(key);
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

struct StringAttrStorage : public mlir::AttributeStorage
{
    llvm::StringRef value;

    StringAttrStorage(llvm::StringRef value, mlir::Type type) : mlir::AttributeStorage(type), value(value) {}

    using KeyTy = std::pair<llvm::StringRef, mlir::Type>;

    bool operator==(KeyTy key) const
    {
        return key.first == value && key.second == getType();
    }

    static llvm::hash_code hashKey(KeyTy key)
    {
        return llvm::hash_value(key);
    }

    static StringAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy value)
    {
        value.first = allocator.copyInto(value.first);
        return new (allocator.allocate<StringAttrStorage>()) StringAttrStorage(value.first, value.second);
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

class StringAttr : public mlir::Attribute::AttrBase<StringAttr, mlir::Attribute, detail::StringAttrStorage>
{
public:
    using Base::Base;

    static StringAttr get(mlir::MLIRContext* context, llvm::StringRef value)
    {
        return Base::get(context, value, StringType::get(context));
    }

    llvm::StringRef getValue()
    {
        return getImpl()->value;
    }
};

class ListAttr : public mlir::Attribute::AttrBase<ListAttr, mlir::Attribute, detail::ListStorage>
{
public:
    using Base::Base;

    static ListAttr get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value)
    {
        return Base::get(context, value, ListType::get(context));
    }

    llvm::ArrayRef<mlir::Attribute> getValue()
    {
        return getImpl()->value;
    }
};

class TupleAttr : public mlir::Attribute::AttrBase<TupleAttr, mlir::Attribute, detail::ListStorage>
{
public:
    using Base::Base;

    static TupleAttr get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value)
    {
        return Base::get(context, value, TupleType::get(context));
    }

    llvm::ArrayRef<mlir::Attribute> getValue()
    {
        return getImpl()->value;
    }
};

class FixedTupleAttr : public mlir::Attribute::AttrBase<FixedTupleAttr, mlir::Attribute, detail::ListStorage>
{
public:
    using Base::Base;

    static FixedTupleAttr get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value)
    {
        std::vector<mlir::Type> types;
        std::transform(value.begin(), value.end(), std::back_inserter(types),
                       [](mlir::Attribute attr) { return attr.getType(); });
        return Base::get(context, value, FixedTupleType::get(types));
    }

    llvm::ArrayRef<mlir::Attribute> getValue()
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
        return Base::get(context, values, SetType::get(context));
    }

    static DictAttr getAlreadySorted(mlir::MLIRContext* context,
                                     llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
    {
        return Base::get(context, value, SetType::get(context));
    }

    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> getValue()
    {
        return getImpl()->value;
    }
};

} // namespace pylir::Dialect
