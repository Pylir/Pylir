//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "PlatformABI.hpp"

namespace pylir {

// Keep in sync with PylirGC.cpp
constexpr unsigned REF_ADDRESS_SPACE = 1;

/// MLIR Dialect conversion type converter, inheriting LLVM type conversions and
/// adding type conversions from Pylir types to LLVM. Also includes various
/// other functions related to translating Pylir types to LLVM.
class PylirTypeConverter : public mlir::LLVMTypeConverter {
  mlir::LLVM::LLVMPointerType m_objectPtrType;
  std::unique_ptr<pylir::PlatformABI> m_cabi;
  mlir::StringAttr m_rootSection;
  mlir::StringAttr m_collectionSection;
  mlir::StringAttr m_constantSection;
  llvm::DenseMap<mlir::Attribute, pylir::Mem::LayoutType> m_layoutTypeCache;

public:
  PylirTypeConverter(mlir::MLIRContext* context, const llvm::Triple& triple,
                     llvm::DataLayout dataLayout,
                     mlir::DataLayout&& mlirDataLayout);

  /// Returns the LLVM representation for 'object', optionally with the given
  /// slot count.
  mlir::LLVM::LLVMStructType
  getPyObjectType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for 'function', optionally with the given
  /// slot count and closure argument types.
  mlir::LLVM::LLVMStructType
  getPyFunctionType(std::optional<unsigned> slotSize = {},
                    mlir::TypeRange closureArgsTypes = {});

  /// Returns the number of bytes 'closureArgsTypes' occupy within a converted
  /// 'PyFunction'.
  unsigned getClosureArgsBytes(mlir::TypeRange closureArgsTypes);

  /// Returns the LLVM representation for a 'tuple', optionally of the given
  /// length.
  mlir::LLVM::LLVMStructType
  getPyTupleType(std::optional<unsigned> length = {});

  /// Returns the LLVM representation for 'list', optionally with the given slot
  /// count.
  mlir::LLVM::LLVMStructType
  getPyListType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for a buffer component, used within 'dict'
  /// and 'str'.
  mlir::LLVM::LLVMStructType getBufferComponent();

  /// Returns the LLVM representation for 'dict', optionally with the given slot
  /// count.
  mlir::LLVM::LLVMStructType
  getPyDictType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for 'str', optionally with the given slot
  /// count.
  mlir::LLVM::LLVMStructType
  getPyStringType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for a 'mp_int' from libtommath.
  mlir::LLVM::LLVMStructType getMPInt();

  /// Returns the LLVM representation for 'int', optionally with the given slot
  /// count.
  mlir::LLVM::LLVMStructType
  getPyIntType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for 'float', optionally with the given
  /// slot count.
  mlir::LLVM::LLVMStructType
  getPyFloatType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for the Itanium ABI's unwind exception
  /// object for the compilation target.
  mlir::Type getUnwindHeaderType() {
    return m_cabi->getUnwindExceptionHeader(&getContext());
  }

  /// Returns the LLVM representation for 'BaseException', optionally with the
  /// given slot count.
  mlir::LLVM::LLVMStructType
  getPyBaseExceptionType(std::optional<unsigned> slotSize = {});

  /// Returns the LLVM representation for 'type', optionally with the given
  /// slots.
  mlir::LLVM::LLVMStructType
  getPyTypeType(std::optional<unsigned> slotSize = {});

  /// Maps from 'pyMem's layout type enum to the corresponding LLVM type
  /// representation, optionally with the given slot count.
  mlir::LLVM::LLVMStructType
  mapLayoutTypeToLLVM(pylir::Mem::LayoutType builtinsName,
                      std::optional<unsigned> slotCount = {}) {
    switch (builtinsName) {
    case pylir::Mem::LayoutType::Object: return getPyObjectType(slotCount);
    case pylir::Mem::LayoutType::Type: return getPyTypeType(slotCount);
    case pylir::Mem::LayoutType::Float: return getPyFloatType(slotCount);
    case pylir::Mem::LayoutType::Function: return getPyFunctionType(slotCount);
    case pylir::Mem::LayoutType::Tuple: return getPyTupleType(slotCount);
    case pylir::Mem::LayoutType::List: return getPyListType(slotCount);
    case pylir::Mem::LayoutType::String: return getPyStringType(slotCount);
    case pylir::Mem::LayoutType::Dict: return getPyDictType(slotCount);
    case pylir::Mem::LayoutType::Int: return getPyIntType(slotCount);
    case pylir::Mem::LayoutType::BaseException:
      return getPyBaseExceptionType(slotCount);
    }
    PYLIR_UNREACHABLE;
  }

  /// Returns the layout type of a given 'py' attribute type object.
  /// This is a simple wrapper around 'pylir::Mem::getLayoutType'. See its
  /// description for more details.
  std::optional<pylir::Mem::LayoutType> getLayoutType(mlir::Attribute attr);

  /// Returns the LLVM type representation for 'objectAttr'. The type has the
  /// required size and storage to store 'objectAttr', including its slots.
  mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttrInterface objectAttr);

  /// Returns the target platform ABI used.
  pylir::PlatformABI& getPlatformABI() const {
    return *m_cabi;
  }

  /// Returns the name of the root section. The root section is a section for
  /// all translated 'py.globalHandle' global variables. These are collected
  /// into an array during code generation and used by the runtime to find still
  /// referenced objects.
  mlir::StringAttr getRootSection() const {
    return m_rootSection;
  }

  /// Returns the name of the collection section. The collection is the section
  /// for all translated 'py.globalValue' variables that are not const and may
  /// therefore contain references to GC allocated objects. These are collected
  /// into an array during code generation and used by the runtime to find still
  /// referenced objects.
  mlir::StringAttr getCollectionSection() const {
    return m_collectionSection;
  }

  /// Returns the name of the constant sections. The constant section is the
  /// section for all translated const 'py.globalValue' variables and
  /// 'py.constant' attributes. While it cannot contain references to any GC
  /// allocated objects, objects within this section must still be known to the
  /// runtime, to avoid marking by the GC and to prune the object graph.
  mlir::StringAttr getConstantSection() const {
    return m_constantSection;
  }
};
} // namespace pylir
