//  Copyright 2022 Markus Böck
//
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

/// Casts the attribute to the given 'T'. If the attribute is a symbol reference it uses 'op' to find the nearest
/// 'py.globalValue' and return and cast its initialize to 'T' instead. If the 'py.globalValue' that is found is not
/// 'const' and 'onlyConstGlobal' is true, a null value is returned.
/// If the cast does not succeed or the 'attr' passed in is a null value, a null value is also returned.
template <class T = ObjectAttrInterface>
T resolveValue(mlir::Operation* op, mlir::Attribute attr, bool onlyConstGlobal = true)
{
    auto ref = attr.dyn_cast_or_null<mlir::SymbolRefAttr>();
    if (!ref)
    {
        return attr.dyn_cast_or_null<T>();
    }
    auto value = mlir::SymbolTable::lookupNearestSymbolFrom<GlobalValueOp>(op, ref);
    // TODO: This 'if' is nasty workaround to make PylirToLLVM not crash. Reason being that dialect conversion attempts
    //       to also do folding to legalize an operation. Since we re mid dialect conversion however, the GlobalValueOp
    //       may have already been converted to LLVM and erased, hence it does not exist anymore. By returning nullptr
    //       we ought to fail gracefully.
    if (!value)
    {
        return nullptr;
    }
    if (!value.getConstant() && onlyConstGlobal)
    {
        return nullptr;
    }
    return value.getInitializerAttr().template dyn_cast_or_null<T>();
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
/// 'context' is used for any symbol lookups to find the nearest symbol table.
BuiltinMethodKind getHashFunction(ObjectAttrInterface attribute, mlir::Operation* context);

} // namespace pylir::Py
