//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace pylir {
/// Abstract base class abstracting the C ABI details of a target platform. It
/// is mainly used to declare and call functions from our runtime which exports
/// its API as C ABI functions. It also offers methods for querying information
/// about types on the target platform.
class PlatformABI {
  mlir::DataLayout m_dataLayout;

public:
  /// Creates a new PlatformABI with the given MLIR data layout. The data layout
  /// is used to query any size and alignment infos of types.
  explicit PlatformABI(mlir::DataLayout&& dataLayout)
      : m_dataLayout(std::move(dataLayout)) {}

  virtual ~PlatformABI() = default;

  PlatformABI(const PlatformABI&) = delete;
  PlatformABI& operator=(const PlatformABI&) = delete;
  PlatformABI(PlatformABI&&) = delete;
  PlatformABI& operator=(PlatformABI&&) = delete;

  /// Returns the size of a type in bytes. Must be an LLVM type or any other
  /// type implementing 'DataLayoutTypeInterface'.
  std::size_t getSizeOf(mlir::Type type) const;

  /// Returns the alignment requirement of a type in bytes. Must be an LLVM type
  /// or any other type implementing 'DataLayoutTypeInterface'.
  std::size_t getAlignOf(mlir::Type type) const;

  /// Function used to declare a C ABI function and do the required
  /// transformations for the C ABI. The function consists of a possibly void
  /// 'returnType', a list of parameter types and is given the name 'name'.
  /// 'builder' is used for the insertion point of where the function
  /// declaration should be put. The function op is also returned. Functions
  /// returned by this method are not allowed to be called directly. One has to
  /// use 'callFunc' as it has to do required transformations for the return
  /// value and arguments.
  ///
  /// Note: The interface here is currently insufficient for actually modelling
  /// all details of the C ABI. In particular, it currently only operates on
  /// LLVM and builtin types, which does not have all information about a C
  /// type. One example is that ABIs often require either sign extension or
  /// unsigned extension of specific integer types which we currently cannot
  /// model.
  virtual mlir::LLVM::LLVMFuncOp
  declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
              mlir::Type returnType, llvm::StringRef name,
              mlir::TypeRange parameterTypes) = 0;

  /// Function used to call a function previously declared with 'declareFunc'.
  /// Returns the result of the call if the callee has a return value.
  virtual mlir::Value callFunc(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::LLVM::LLVMFuncOp func,
                               mlir::ValueRange arguments) = 0;

  /// Returns the integer type corresponding to 'int' on the target platform.
  virtual mlir::Type getInt(mlir::MLIRContext* context) const = 0;

  /// Returns the type corresponding to the platforms unwind exception header in
  /// the Itanium ABI on the target platform.
  virtual mlir::Type
  getUnwindExceptionHeader(mlir::MLIRContext* context) const = 0;

  /// Returns the integer type corresponding to 'size_t' on the target platform.
  virtual mlir::Type getSizeT(mlir::MLIRContext* context) const = 0;
};
} // namespace pylir
