//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>

#include "PylirPyTypes.hpp"

namespace pylir::Py
{
/// Possible return values of `TypeRefineableInterface::refineTypes`
enum class TypeRefineResult
{
    Failure,     /// Failed to compute all the resulting types.
    Approximate, /// Only managed to compute an approximate result. The runtime type may be more precise.
    Success,     /// Successfully computed the resulting types.
};

/// The Lattice of type deduction analysis'. It's top value is the NULL value. The bottom value is not directly
/// represented in the lattice, but is indicated by the absence of a mapped TypeAttrUnion in the respective maps.
/// Additionally, `!py.unbound` server as the bottom, but for types only.
class TypeAttrUnion : private llvm::PointerUnion<mlir::Attribute, pylir::Py::ObjectTypeInterface>
{
    using Base = PointerUnion<mlir::Attribute, pylir::Py::ObjectTypeInterface>;

    friend struct llvm::DenseMapInfo<TypeAttrUnion>;
    friend struct llvm::PointerLikeTypeTraits<TypeAttrUnion>;

    // NOLINTNEXTLINE(google-explicit-constructor)
    TypeAttrUnion(Base base) : Base(base) {}

public:
    TypeAttrUnion() = default;

    /*implicit*/ TypeAttrUnion(std::nullptr_t) : Base(nullptr) {}

    /*implicit*/ TypeAttrUnion(ObjectTypeInterface type) : Base(type) {}

    /*implicit*/ TypeAttrUnion(mlir::Attribute attr) : Base(attr) {}

    template <class... Args>
    [[nodiscard]] bool isa() const
    {
        // Can't check for ObjectTypeInterface statically.
        static_assert(((std::is_base_of_v<mlir::Attribute, Args> || std::is_base_of_v<mlir::Type, Args>)&&...));
        return (
            [this](
                auto type) -> bool
                                  {
                                      using T = typename decltype(type)::argument_type;
                                      if constexpr (std::is_base_of_v<mlir::Attribute, T>)
                                      {
                                          return Base::dyn_cast<mlir::Attribute>().isa_and_nonnull<T>();
                                      }
                                      else
                                      {
                                          return Base::dyn_cast<pylir::Py::ObjectTypeInterface>().isa_and_nonnull<T>();
                                      }
                                  }(llvm::identity<Args>{})
                                  || ...);
    }

    template <class... Args>
    [[nodiscard]] bool isa_and_nonnull() const
    {
        if (!*this)
        {
            return false;
        }
        return isa<Args...>();
    }

    template <class T>
    [[nodiscard]] T cast() const
    {
        static_assert(std::is_base_of_v<mlir::Attribute, T> || std::is_base_of_v<mlir::Type, T>);
        if constexpr (std::is_base_of_v<mlir::Attribute, T>)
        {
            return get<mlir::Attribute>().cast<T>();
        }
        else
        {
            return get<pylir::Py::ObjectTypeInterface>().template cast<T>();
        }
    }

    template <class T>
    [[nodiscard]] T dyn_cast() const
    {
        return isa<T>() ? cast<T>() : nullptr;
    }

    template <class T>
    [[nodiscard]] T dyn_cast_or_null() const
    {
        return isa_and_nonnull<T>() ? cast<T>() : nullptr;
    }

    pylir::Py::TypeAttrUnion join(pylir::Py::TypeAttrUnion rhs);

    using Base::operator bool;

    bool operator==(const TypeAttrUnion& rhs) const
    {
        return static_cast<const Base&>(*this) == static_cast<const Base&>(rhs);
    }

    void dump();
};
} // namespace pylir::Py

template <>
struct llvm::PointerLikeTypeTraits<pylir::Py::TypeAttrUnion>
{
    static void* getAsVoidPointer(const pylir::Py::TypeAttrUnion& p)
    {
        return p.getOpaqueValue();
    }

    static pylir::Py::TypeAttrUnion getFromVoidPointer(void* p)
    {
        return pylir::Py::TypeAttrUnion::getFromOpaqueValue(p);
    }

    static constexpr int NumLowBitsAvailable =
        PointerLikeTypeTraits<pylir::Py::TypeAttrUnion::Base>::NumLowBitsAvailable;
};

template <>
struct llvm::DenseMapInfo<pylir::Py::TypeAttrUnion>
{
    static inline pylir::Py::TypeAttrUnion getEmptyKey()
    {
        return llvm::DenseMapInfo<pylir::Py::TypeAttrUnion::Base>::getEmptyKey();
    }

    static inline pylir::Py::TypeAttrUnion getTombstoneKey()
    {
        return llvm::DenseMapInfo<pylir::Py::TypeAttrUnion::Base>::getTombstoneKey();
    }

    static inline unsigned getHashValue(const pylir::Py::TypeAttrUnion& value)
    {
        return llvm::DenseMapInfo<pylir::Py::TypeAttrUnion::Base>::getHashValue(value);
    }

    static inline bool isEqual(const pylir::Py::TypeAttrUnion& lhs, const pylir::Py::TypeAttrUnion& rhs)
    {
        return lhs == rhs;
    }
};

namespace pylir::Py
{
inline llvm::hash_code hash_value(TypeAttrUnion value)
{
    return llvm::DenseMapInfo<TypeAttrUnion>::getHashValue(value);
}
} // namespace pylir::Py

#include "pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.h.inc"
