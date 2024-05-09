//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "WinX64.hpp"

#include <mlir/IR/Builders.h>

#include <pylir/Support/Macros.hpp>

namespace {
bool isLegalIntegerSize(std::size_t size) {
  return size == 1 || size == 2 || size == 4 || size == 8;
}
} // namespace

mlir::LLVM::LLVMFuncOp
pylir::WinX64::declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Type returnType, llvm::StringRef name,
                           mlir::TypeRange parameterTypes) {
  mlir::Type retType = returnType;
  llvm::SmallVector<mlir::Type> argumentTypes;
  argumentTypes.reserve(parameterTypes.size());

  Adjustments adjustments;
  adjustments.originalRetType = returnType;
  adjustments.arguments.resize(parameterTypes.size());

  if (mlir::isa<mlir::LLVM::LLVMStructType>(returnType)) {
    auto size = getSizeOf(returnType);
    if (isLegalIntegerSize(size)) {
      adjustments.returnType = IntegerRegister;
      retType = builder.getIntegerType(size * 8);
    } else {
      adjustments.returnType = PointerToTemporary;
      retType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
      argumentTypes.push_back(builder.getType<mlir::LLVM::LLVMPointerType>());
    }
  }

  for (std::size_t i = 0; i < parameterTypes.size(); i++) {
    if (!mlir::isa<mlir::LLVM::LLVMStructType>(parameterTypes[i])) {
      argumentTypes.emplace_back(parameterTypes[i]);
      continue;
    }
    auto size = getSizeOf(parameterTypes[i]);
    if (isLegalIntegerSize(size)) {
      argumentTypes.push_back(builder.getIntegerType(size * 8));
      adjustments.arguments[i] = IntegerRegister;
    } else {
      argumentTypes.push_back(builder.getType<mlir::LLVM::LLVMPointerType>());
      adjustments.arguments[i] = PointerToTemporary;
    }
  }

  auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(
      loc, name, mlir::LLVM::LLVMFunctionType::get(retType, argumentTypes));
  if (adjustments.returnType == PointerToTemporary) {
    funcOp.setArgAttrs(
        0, {builder.getNamedAttr(mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                                 builder.getUnitAttr()),
            builder.getNamedAttr("llvm.sret", builder.getUnitAttr())});
  }
  m_adjustments.insert({funcOp, adjustments});
  return funcOp;
}

mlir::Value pylir::WinX64::callFunc(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::LLVM::LLVMFuncOp func,
                                    mlir::ValueRange arguments) {
  auto result = m_adjustments.find(func);
  PYLIR_ASSERT(result != m_adjustments.end());
  auto& adjustments = result->second;
  mlir::LLVM::AllocaOp returnSlot;

  llvm::SmallVector<mlir::Value> transformedArgs;

  std::size_t paramBegin = 0;
  if (adjustments.returnType == PointerToTemporary) {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&builder.getBlock()->getParent()->front());
    paramBegin = 1;
    auto one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    returnSlot = builder.create<mlir::LLVM::AllocaOp>(
        loc, func.getFunctionType().getParams().front(),
        adjustments.originalRetType, one, 1);
    transformedArgs.push_back(returnSlot);
  }

  for (std::size_t i = 0; i < arguments.size(); i++) {
    switch (adjustments.arguments[i]) {
    case Nothing: transformedArgs.push_back(arguments[i]); break;
    case IntegerRegister: {
      auto integerPointerType = builder.getType<mlir::LLVM::LLVMPointerType>();
      if (auto load = arguments[i].getDefiningOp<mlir::LLVM::LoadOp>()) {
        transformedArgs.push_back(builder.create<mlir::LLVM::LoadOp>(
            loc, arguments[i].getType(), load.getAddr()));
        break;
      }
      auto one = builder.create<mlir::LLVM::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      mlir::Type convertedType =
          func.getFunctionType().getParams()[paramBegin + i];
      auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
          loc, integerPointerType, convertedType, one, 1);
      builder.create<mlir::LLVM::StoreOp>(loc, arguments[i], tempAlloca);
      transformedArgs.push_back(
          builder.create<mlir::LLVM::LoadOp>(loc, convertedType, tempAlloca));
      break;
    }
    case PointerToTemporary: {
      auto one = builder.create<mlir::LLVM::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
          loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
          arguments[i].getType(), one, 1);
      transformedArgs.push_back(tempAlloca);
      if (auto load = arguments[i].getDefiningOp<mlir::LLVM::LoadOp>()) {
        auto falseConstant = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI1Type(), builder.getBoolAttr(false));
        auto null =
            builder.create<mlir::LLVM::ZeroOp>(loc, tempAlloca.getType());
        auto gep = builder.create<mlir::LLVM::GEPOp>(
            loc, null.getType(), tempAlloca.getElemType(), null,
            mlir::ValueRange{one});
        auto size = builder.create<mlir::LLVM::PtrToIntOp>(
            loc, builder.getI64Type(), gep);
        builder.create<mlir::LLVM::MemcpyOp>(loc, tempAlloca, load.getAddr(),
                                             size, falseConstant);
        break;
      }

      builder.create<mlir::LLVM::StoreOp>(loc, arguments[i], tempAlloca);
      break;
    }
    }
  }

  auto call = builder.create<mlir::LLVM::CallOp>(loc, func, arguments);

  switch (adjustments.returnType) {
  case Nothing:
    if (call.getNumResults() == 0)
      return {};

    return call.getResult();
  case IntegerRegister: {
    auto one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
        loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
        adjustments.originalRetType, one, 1);
    builder.create<mlir::LLVM::StoreOp>(loc, call.getResult(), tempAlloca);
    return builder.create<mlir::LLVM::LoadOp>(loc, adjustments.originalRetType,
                                              tempAlloca);
  }
  case PointerToTemporary:
    return builder.create<mlir::LLVM::LoadOp>(loc, returnSlot.getElemType(),
                                              returnSlot);
  }
  PYLIR_UNREACHABLE;
}

mlir::Type
pylir::WinX64::getUnwindExceptionHeader(mlir::MLIRContext* context) const {
  auto i64 = mlir::IntegerType::get(context, 64);
  // See _Unwind_Exception in unwind.h
  return mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {
          /*exception_class*/ i64,
          /*exception_cleanup*/ mlir::LLVM::LLVMPointerType::get(context),
          /*private_*/ mlir::LLVM::LLVMArrayType::get(i64, 6),
      });
}
