//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "X86_64.hpp"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Support/Variant.hpp>

namespace {
llvm::SmallVector<mlir::Type> flatten(mlir::Type type) {
  llvm::SmallVector<mlir::Type> result;
  if (auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    for (auto iter : structType.getBody()) {
      auto temp = flatten(iter);
      result.insert(result.end(), temp.begin(), temp.end());
    }
  } else if (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
    auto temp = flatten(arrayType.getElementType());
    for (std::size_t i = 0; i < arrayType.getNumElements(); i++) {
      result.insert(result.end(), temp.begin(), temp.end());
    }
  } else {
    result = {type};
  }
  return result;
}

} // namespace

std::tuple<pylir::X86_64::Adjustments::Arg, mlir::Type, mlir::Type>
pylir::X86_64::flattenSingleArg(mlir::Type type, std::uint8_t* takenIntegers,
                                std::uint8_t* takenFloats) {
  constexpr std::uint8_t availableIntegerRegisters = 6;
  constexpr std::uint8_t availableFloatingPointRegisters = 8;
  Adjustments::Arg dest;
  std::size_t retIndex = 0;
  std::array<mlir::Type, 2> ret = {};

  std::uint8_t takenIntegerRegisters = takenIntegers ? *takenIntegers : 0;
  std::uint8_t takenFloatingPointRegisters = takenFloats ? *takenFloats : 0;
  const auto flat = flatten(type);
  const auto* iter = flat.begin();
  while (iter != flat.end()) {
    const auto* const begin = iter;
    bool encounteredInteger = false;
    std::size_t size = 0;
    std::size_t currentAlignment = 0;
    while (size < 8 && iter != flat.end()) {
      auto alignment = getSizeOf(*iter);
      const auto temp = llvm::alignTo(size, alignment);
      if (temp >= 8)
        break;

      size = temp;
      if (mlir::isa<mlir::IntegerType, mlir::LLVM::LLVMPointerType>(*iter)) {
        encounteredInteger = true;
      }
      // Special case X86_FP80Ty if ever used here
      currentAlignment = std::max(currentAlignment, alignment);
      size += alignment;
      iter++;
    }
    size = llvm::alignTo(size, currentAlignment);
    if (encounteredInteger) {
      // We encountered at least one integer therefore even if a floating point
      // type was in there it's gotta go into a integer register
      if (takenIntegerRegisters >= availableIntegerRegisters) {
        if (mlir::isa<mlir::LLVM::LLVMStructType>(type))
          return {OnStack{}, type, nullptr};

        return {Unchanged{}, type, nullptr};
      }

      takenIntegerRegisters++;
      if (mlir::isa<mlir::LLVM::LLVMStructType>(type) &&
          !std::holds_alternative<MultipleArgs>(dest))
        dest = MultipleArgs{};

      if (mlir::isa<mlir::LLVM::LLVMStructType>(type))
        ret[retIndex++] = mlir::IntegerType::get(type.getContext(), size * 8);
      else
        ret[retIndex++] = type;

      if (auto* multiArgs = std::get_if<MultipleArgs>(&dest))
        multiArgs->size++;

      continue;
    }
    if (std::distance(begin, iter) == 2 &&
        mlir::isa<mlir::Float32Type>(*begin) &&
        mlir::isa<mlir::Float32Type>(*(begin + 1))) {
      // Two floats can be packed as a single 64 bit value int a xmm register.
      // This is represented as a vector in LLVM IR
      if (takenFloatingPointRegisters >= availableFloatingPointRegisters) {
        if (mlir::isa<mlir::LLVM::LLVMStructType>(type))
          return {OnStack{}, type, nullptr};

        return {Unchanged{}, type, nullptr};
      }
      takenFloatingPointRegisters++;
      if (!std::holds_alternative<MultipleArgs>(dest))
        dest = MultipleArgs{};

      ret[retIndex++] = mlir::LLVM::LLVMFixedVectorType::get(*begin, 2);
      pylir::get<MultipleArgs>(dest).size++;
      continue;
    }

    PYLIR_ASSERT(std::distance(begin, iter) == 1);
    // Must be a floating point type because if it were integer it would have
    // taken the encounteredInteger branch above
    if (takenFloatingPointRegisters >= availableFloatingPointRegisters) {
      if (mlir::isa<mlir::LLVM::LLVMStructType>(type))
        return {OnStack{}, type, nullptr};

      return {Unchanged{}, type, nullptr};
    }
    takenFloatingPointRegisters++;
    if (mlir::isa<mlir::LLVM::LLVMStructType>(type) &&
        !std::holds_alternative<MultipleArgs>(dest))
      dest = MultipleArgs{};

    ret[retIndex++] = *begin;
    if (auto* multiArgs = std::get_if<MultipleArgs>(&dest))
      multiArgs->size++;
  }
  if (takenFloats)
    *takenFloats = takenFloatingPointRegisters;

  if (takenIntegers)
    *takenIntegers = takenIntegerRegisters;

  return {dest, ret[0], ret[1]};
}

mlir::LLVM::LLVMFuncOp
pylir::X86_64::declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Type returnType, llvm::StringRef name,
                           mlir::TypeRange parameterTypes) {
  Adjustments adjustments;
  adjustments.arguments.reserve(parameterTypes.size());
  adjustments.originalRetType = returnType;

  mlir::Type retType = returnType;
  llvm::SmallVector<mlir::Type> argumentTypes;
  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(returnType)) {
    auto size = getSizeOf(returnType);
    if (size > 16) {
      adjustments.returnType = PointerToTemporary{};
      argumentTypes.push_back(builder.getType<mlir::LLVM::LLVMPointerType>());
      retType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    } else {
      bool wasStruct = mlir::isa<mlir::LLVM::LLVMStructType>(returnType);
      std::pair<mlir::Type, mlir::Type> types;
      std::tie(std::ignore, types.first, types.second) =
          flattenSingleArg(returnType);
      if (types.second)
        retType = mlir::LLVM::LLVMStructType::getLiteral(
            builder.getContext(), {types.first, types.second});
      else
        retType = types.first;

      if (wasStruct)
        adjustments.returnType = Flattened{};
    }
  }

  std::uint8_t takenIntegerRegisters = 0;
  std::uint8_t takenFloatingPointRegisters = 0;
  for (auto inputType : parameterTypes) {
    if (getSizeOf(inputType) > 16) {
      adjustments.arguments.emplace_back(OnStack{});
      argumentTypes.push_back(builder.getType<mlir::LLVM::LLVMPointerType>());
      continue;
    }
    std::pair<mlir::Type, mlir::Type> types;
    Adjustments::Arg dest;
    std::tie(dest, types.first, types.second) = flattenSingleArg(
        inputType, &takenIntegerRegisters, &takenFloatingPointRegisters);
    adjustments.arguments.push_back(dest);
    if (std::holds_alternative<OnStack>(dest)) {
      argumentTypes.push_back(builder.getType<mlir::LLVM::LLVMPointerType>());
    } else if (std::holds_alternative<MultipleArgs>(dest)) {
      argumentTypes.push_back(types.first);
      if (types.second)
        argumentTypes.push_back(types.second);

    } else {
      argumentTypes.push_back(inputType);
    }
  }

  auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(
      loc, name, mlir::LLVM::LLVMFunctionType::get(retType, argumentTypes));
  std::size_t argStart = 0;
  if (std::holds_alternative<PointerToTemporary>(adjustments.returnType)) {
    argStart = 1;
    funcOp.setArgAttrs(
        0, {builder.getNamedAttr(mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                                 builder.getUnitAttr()),
            builder.getNamedAttr("llvm.sret", builder.getUnitAttr())});
  }
  std::size_t origArgI = 0;
  for (std::size_t i = argStart; i < funcOp.getNumArguments(); origArgI++) {
    pylir::match(
        adjustments.arguments[origArgI], [&](Unchanged) { i++; },
        [&](OnStack) {
          auto exit = llvm::make_scope_exit([&] { i++; });
          funcOp.setArgAttrs(
              i,
              {builder.getNamedAttr("llvm.byval", builder.getUnitAttr()),
               builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                                    builder.getI64IntegerAttr(8))});
        },
        [&](MultipleArgs multipleArgs) { i += multipleArgs.size; });
  }
  m_adjustments.insert({funcOp, adjustments});
  return funcOp;
}

mlir::Value pylir::X86_64::callFunc(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::LLVM::LLVMFuncOp func,
                                    mlir::ValueRange arguments) {
  auto result = m_adjustments.find(func);
  PYLIR_ASSERT(result != m_adjustments.end());
  auto& adjustments = result->second;
  mlir::LLVM::AllocaOp returnSlot;

  llvm::SmallVector<mlir::Value> transformedArgs;

  if (std::holds_alternative<PointerToTemporary>(adjustments.returnType)) {
    auto one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    returnSlot = builder.create<mlir::LLVM::AllocaOp>(
        loc, func.getFunctionType().getParams().front(),
        adjustments.originalRetType, one, 1);
    transformedArgs.push_back(returnSlot);
  }

  for (auto iter = arguments.begin(); iter != arguments.end(); iter++) {
    auto getPointerToMemory = [&](mlir::Value value) -> mlir::Value {
      if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
        return load.getAddr();

      auto one = builder.create<mlir::LLVM::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
          loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
          (*iter).getType(), one, 1);
      builder.create<mlir::LLVM::StoreOp>(loc, *iter, tempAlloca);
      return tempAlloca;
    };
    pylir::match(
        adjustments.arguments[iter - arguments.begin()],
        [&](Unchanged) { transformedArgs.push_back(*iter); },
        [&](OnStack) { transformedArgs.push_back(getPointerToMemory(*iter)); },
        [&](MultipleArgs multipleArgs) {
          auto address = getPointerToMemory(*iter);
          if (multipleArgs.size == 1) {
            auto paramType =
                func.getFunctionType().getParamType(arguments.size());
            auto casted = builder.create<mlir::LLVM::BitcastOp>(
                loc, builder.getType<mlir::LLVM::LLVMPointerType>(), address);
            transformedArgs.push_back(
                builder.create<mlir::LLVM::LoadOp>(loc, paramType, casted));
            return;
          }
          auto firstType =
              func.getFunctionType().getParamType(arguments.size());
          auto secondType =
              func.getFunctionType().getParamType(arguments.size() + 1);
          auto zero = builder.create<mlir::LLVM::ConstantOp>(
              loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
          auto one = builder.create<mlir::LLVM::ConstantOp>(
              loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
          auto firstValue = builder.create<mlir::LLVM::GEPOp>(
              loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
              (*iter).getType(), address, mlir::ValueRange{zero, zero});
          auto secondValue = builder.create<mlir::LLVM::GEPOp>(
              loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
              (*iter).getType(), address, mlir::ValueRange{zero, one});
          transformedArgs.push_back(
              builder.create<mlir::LLVM::LoadOp>(loc, firstType, firstValue));
          transformedArgs.push_back(
              builder.create<mlir::LLVM::LoadOp>(loc, secondType, secondValue));
        });
  }

  auto call = builder.create<mlir::LLVM::CallOp>(loc, func, arguments);
  return pylir::match(
      adjustments.returnType,
      [&](Unchanged) -> mlir::Value {
        if (call->getNumResults() > 0)
          return call.getResult();

        return {};
      },
      [&](PointerToTemporary) -> mlir::Value {
        return builder.create<mlir::LLVM::LoadOp>(
            loc, adjustments.originalRetType, returnSlot);
      },
      [&](Flattened) -> mlir::Value {
        auto one = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
            loc, builder.getType<mlir::LLVM::LLVMPointerType>(),
            adjustments.originalRetType, one, 1);
        builder.create<mlir::LLVM::StoreOp>(loc, call.getResult(), tempAlloca);
        return builder.create<mlir::LLVM::LoadOp>(
            loc, adjustments.originalRetType, tempAlloca);
      });
}

mlir::Type
pylir::X86_64::getUnwindExceptionHeader(mlir::MLIRContext* context) const {
  auto i64 = mlir::IntegerType::get(context, 64);
  // See _Unwind_Exception in unwind.h
  return mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {
          /*exception_class*/ i64,
          /*exception_cleanup*/ mlir::LLVM::LLVMPointerType::get(context),
          /*private_1*/ i64,
          /*private_2*/ i64,
      });
}
