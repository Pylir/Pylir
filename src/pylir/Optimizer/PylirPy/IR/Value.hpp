//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/SymbolTable.h>

#include "PylirPyAttrInterfaces.hpp"
#include "PylirPyOps.hpp"

namespace pylir::Py {
/// Returns the type of the value. This may either be a value referring to the
/// type object or an attribute that is the type object. This operation may also
/// fail in which case it is a null value.
mlir::OpFoldResult getTypeOf(mlir::Value value);

/// Returns whether the value is definitely bound, unbound or unknown. If the
/// optional does not have a value, it is unknown whether it's bound or not,
/// otherwise the optional contains whether the value is unbound.
std::optional<bool> isUnbound(mlir::Value value);

/// Casts the attribute to the given 'T'. It differs from 'dyn_cast' in the case
/// that the attribute is a 'RefAttr'. It then uses the referred to
/// 'py.globalValue' and attempts to cast its initializer to 'T' instead.
///
/// If the 'py.globalValue' used is not 'constant' and 'onlyConstGlobal' is
/// true, a null value is returned. By default, 'onlyConstGlobal' is set to
/// true, unless 'T' is an instance of 'ImmutableAttr' or 'IntAttrInterface'.
///
/// The rationale is that if one of these attributes is requested, one is almost
/// certainly going to only read from the immutable value parts of the
/// attribute, even if the 'py.globalValue' may not be 'constant'.
///
/// Nevertheless, one has to make sure **NOT TO READ FROM THE SLOTS OF AN
/// IMMUTABLE ATTR** in such a case, as they may not be constant! Either
/// 'ref_cast' to 'ObjectAttrInterface' instead or explicitly set
/// 'onlyConstGlobal' to true.
///
/// If the cast does not succeed a null value is also returned.
/// These are intentionally snake_case to follow 'dyn_cast's naming convention.
template <class T>
T ref_cast(mlir::Attribute attr,
           bool onlyConstGlobal = !std::is_base_of_v<ImmutableAttr<T>, T>) {
  if (auto val = attr.dyn_cast<T>()) {
    return val;
  }
  auto ref = attr.dyn_cast<RefAttr>();
  if (!ref || (!ref.getSymbol().getConstant() && onlyConstGlobal)) {
    return nullptr;
  }
  return ref.getSymbol().getInitializerAttr().dyn_cast_or_null<T>();
}

/// Same as 'ref_cast' but returns a null value if 'attr' is null.
/// These are intentionally snake_case to follow 'dyn_cast's naming convention.
template <class T>
T ref_cast_or_null(
    mlir::Attribute attr,
    bool onlyConstGlobal = !std::is_base_of_v<ImmutableAttr<T>, T>) {
  return attr ? ref_cast<T>(attr, onlyConstGlobal) : nullptr;
}

enum class BuiltinMethodKind { Unknown, Str, Int, Float, Object };

/// Given any kind of object, attempts to return the hash function used at
/// runtime to hash that object. If the hash function may change at runtime or a
/// custom one is used that is not known to the compiler, 'Unknown' is returned.
///
/// 'attribute' has to either be a 'RefAttr' or 'ObjectAttrInterface', otherwise
/// the behaviour is undefined.
BuiltinMethodKind getHashFunction(mlir::Attribute attribute);

/// Given any kind of object, attempts to return the __eq__ function used at
/// runtime to compare that object to other objects. If the hash function may
/// change at runtime or a custom one is used that is not known to the compiler,
/// 'Unknown' is returned.
///
/// 'attribute' has to either be a 'RefAttr' or 'ObjectAttrInterface', otherwise
/// the behaviour is undefined.
BuiltinMethodKind getEqualsFunction(mlir::Attribute attribute);

/// Given either a 'RefAttr', 'UnboundAttr' or 'ObjectAttrInterface', returns an
/// attribute suitable for equality comparison. The precise attribute returned
/// is guaranteed to be the same attribute for every input attribute which would
/// compare equal according to '__eq__' at runtime. The precise kind of
/// attribute returned is irrelevant to majority of users but may be required
/// for the textual IR presentation of `DictAttr`. See its description for the
/// concrete attribute kinds by this function.
///
/// This requires that the '__eq__' implementation of the attribute, as
/// determined by 'getEqualsFunction', is known to the compiler. If it is
/// unknown, or the attribute passed in is not considered suitable for the given
/// equality implementation, a null attribute is returned.
mlir::Attribute getCanonicalEqualsForm(mlir::Attribute attribute);

/// Returns whether 'lhs' and 'rhs' are equal according to '__eq__' or an empty
/// optional if unknown as determined by 'getCanonicalEqualsForm' for each of
/// 'lhs' and 'rhs'. Both are required to be one of 'RefAttr', 'UnboundAttr' or
/// 'ObjectAttrInterface'.
std::optional<bool> isEqual(mlir::Attribute lhs, mlir::Attribute rhs);

} // namespace pylir::Py
