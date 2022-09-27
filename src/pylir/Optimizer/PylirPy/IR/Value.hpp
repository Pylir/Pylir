//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>

#include "ObjectAttrInterface.hpp"
#include "PylirPyOps.hpp"

namespace pylir::Py
{
/// Returns the type of the value. This may either be a value referring to the type object or an attribute that is the
/// type object. This operation may also fail in which case it is a null value.
mlir::OpFoldResult getTypeOf(mlir::Value value);

/// Returns whether the value is definitely bound, unbound or unknown. If the optional does not have a value, it is
/// unknown whether it's bound or not, otherwise the optional contains whether the value is unbound.
llvm::Optional<bool> isUnbound(mlir::Value value);

/// Casts the attribute to the given 'T'. It differs from 'dyn_cast' in the case that the attribute is a 'RefAttr'. It
/// then uses the referred to 'py.globalValue' and attempts to cast its initializer to 'T' instead.
///
/// If the 'py.globalValue' used is not 'constant' and 'onlyConstGlobal' is true, a null value is returned.
/// By default, 'onlyConstGlobal' is set to true, unless 'T' is an instance of 'ImmutableAttr' or 'IntAttrInterface'.
///
/// The rationale is that if one of these attributes is requested, one is almost certainly going to only read from the
/// immutable value parts of the attribute, even if the 'py.globalValue' may not be 'constant'.
///
/// Nevertheless, one has to make sure **NOT TO READ FROM THE SLOTS OF AN IMMUTABLE ATTR** in such a case, as they may
/// not be constant! Either 'ref_cast' to 'ObjectAttrInterface' instead or explicitly set 'onlyConstGlobal' to true.
///
/// If the cast does not succeed a null value is also returned.
/// These are intentionally snake_case to follow 'dyn_cast's naming convention.
template <class T>
T ref_cast(mlir::Attribute attr, bool onlyConstGlobal = !std::is_base_of_v<ImmutableAttr<T>, T>)
{
    if (auto val = attr.dyn_cast<T>())
    {
        return val;
    }
    auto ref = attr.dyn_cast<RefAttr>();
    if (!ref || (!ref.getSymbol().getConstant() && onlyConstGlobal))
    {
        return nullptr;
    }
    return ref.getSymbol().getInitializerAttr().dyn_cast_or_null<T>();
}

/// Same as 'ref_cast' but returns a null value if 'attr' is null.
/// These are intentionally snake_case to follow 'dyn_cast's naming convention.
template <class T>
T ref_cast_or_null(mlir::Attribute attr, bool onlyConstGlobal = !std::is_base_of_v<ImmutableAttr<T>, T>)
{
    return attr ? ref_cast<T>(attr, onlyConstGlobal) : nullptr;
}

enum class BuiltinMethodKind
{
    Unknown,
    Str,
    Int,
    Object
};

/// Given any kind of object, attempts to return the hash function used at runtime to hash that object. If the hash
/// function may change at runtime or a custom one is used that is not known to the compiler, 'Unknown' is returned.
BuiltinMethodKind getHashFunction(ObjectAttrInterface attribute);

} // namespace pylir::Py
