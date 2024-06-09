//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>

#include "CodeGenState.hpp"
#include "PylirTypeConverter.hpp"

using namespace mlir;
using namespace pylir::Py;

namespace pylir {
#define GEN_PASS_DEF_CONVERTPYLIRTOLLVMPASS
#include "pylir/Optimizer/Conversion/Passes.h.inc"
} // namespace pylir

namespace {

using namespace pylir;

mlir::Value unrealizedConversion(mlir::OpBuilder& builder, mlir::Value value,
                                 mlir::Type resType) {
  return builder
      .create<mlir::UnrealizedConversionCastOp>(value.getLoc(), resType, value)
      .getResult(0);
}

mlir::Value unrealizedConversion(mlir::OpBuilder& builder, mlir::Value value,
                                 PylirTypeConverter& typeConverter) {
  return builder
      .create<mlir::UnrealizedConversionCastOp>(
          value.getLoc(), typeConverter.convertType(value.getType()), value)
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

template <class T>
struct ConvertPylirOpToLLVMPattern : public mlir::ConvertOpToLLVMPattern<T> {
  explicit ConvertPylirOpToLLVMPattern(PylirTypeConverter& typeConverter,
                                       CodeGenState& codeGenState,
                                       mlir::PatternBenefit benefit = 1)
      : mlir::ConvertOpToLLVMPattern<T>(typeConverter, benefit),
        typeConverter(typeConverter), codeGenState(codeGenState) {}

protected:
  PylirTypeConverter& typeConverter;
  CodeGenState& codeGenState;

  //===--------------------------------------------------------------------===//
  // Model instantiations.
  //===--------------------------------------------------------------------===//

  PyFunctionModel pyFunctionModel(OpBuilder& builder, Value value,
                                  TypeRange closureArgsTypes = {}) const {
    // TODO: Right now we straight up rely on the slot size here matching with
    //  whatever is input in the IR. In the future slots should just be removed
    //  from functions.
    return {builder, value,
            typeConverter.getPyFunctionType(
                std::initializer_list<int>{
#define FUNCTION_SLOT(...) 0,
#include <pylir/Interfaces/Slots.def>
                }
                    .size(),
                closureArgsTypes),
            codeGenState};
  }

#define DEFINE_MODEL_INST(type, name)                            \
  type name(mlir::OpBuilder& builder, mlir::Value value) const { \
    return {builder, value, codeGenState};                       \
  }                                                              \
  static_assert(true)

  DEFINE_MODEL_INST(PyObjectModel, pyObjectModel);
  DEFINE_MODEL_INST(PyListModel, pyListModel);
  DEFINE_MODEL_INST(PyTupleModel, pyTupleModel);
  DEFINE_MODEL_INST(PyDictModel, pyDictModel);
  DEFINE_MODEL_INST(PyIntModel, pyIntModel);
  DEFINE_MODEL_INST(PyFloatModel, pyFloatModel);
  DEFINE_MODEL_INST(PyStringModel, pyStringModel);
  DEFINE_MODEL_INST(PyTypeModel, pyTypeModel);

#undef DEFINE_MODEL_INST
};

struct ConstantOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ConstantOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ConstantOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto value =
        codeGenState.getConstant(op.getLoc(), rewriter, adaptor.getConstant());
    rewriter.replaceOp(op, {value});
    return mlir::success();
  }
};

struct ExternalOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ExternalOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ExternalOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ExternalOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    // Get the global value to make sure it has been constructed.
    // Nothing else has to be done but erasing this op as it is already
    // constructed with external linkage.
    codeGenState.getGlobalValue(rewriter, op.getAttr());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GlobalOpConversion : public ConvertPylirOpToLLVMPattern<Py::GlobalOp> {
  using ConvertPylirOpToLLVMPattern<Py::GlobalOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::GlobalOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::LLVM::Linkage linkage;
    switch (op.getVisibility()) {
    case mlir::SymbolTable::Visibility::Public:
      linkage = mlir::LLVM::linkage::Linkage::External;
      break;
    case mlir::SymbolTable::Visibility::Private:
      linkage = mlir::LLVM::linkage::Linkage::Private;
      break;
    case mlir::SymbolTable::Visibility::Nested: PYLIR_UNREACHABLE;
    }
    auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, typeConverter.convertType(op.getType()), false, linkage,
        op.getName(), mlir::Attribute{},
        typeConverter.getPlatformABI().getAlignOf(
            typeConverter.convertType(op.getType())),
        0, true);
    if (isa<Py::DynamicType>(op.getType()))
      global.setSectionAttr(typeConverter.getRootSection());

    rewriter.setInsertionPointToStart(
        &global.getInitializerRegion().emplaceBlock());
    if (!op.getInitializerAttr()) {
      mlir::Value undef =
          rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), global.getType());
      rewriter.create<mlir::LLVM::ReturnOp>(op.getLoc(), undef);
      return mlir::success();
    }

    llvm::TypeSwitch<mlir::Attribute>(op.getInitializerAttr())
        .Case<mlir::IntegerAttr, mlir::FloatAttr>([&](auto attr) {
          mlir::Value constant = rewriter.create<mlir::LLVM::ConstantOp>(
              op.getLoc(), global.getType(), attr);
          rewriter.create<mlir::LLVM::ReturnOp>(op.getLoc(), constant);
        })
        .Case<Py::ObjectAttrInterface, Py::GlobalValueAttr>([&](auto attr) {
          mlir::Value address =
              codeGenState.getConstant(op.getLoc(), rewriter, attr);
          rewriter.create<mlir::LLVM::ReturnOp>(op.getLoc(), address);
        });

    return mlir::success();
  }
};

struct LoadOpConversion : public ConvertPylirOpToLLVMPattern<Py::LoadOp> {
  using ConvertPylirOpToLLVMPattern<Py::LoadOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::LoadOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), rewriter.getType<mlir::LLVM::LLVMPointerType>(),
        op.getGlobalAttr());
    rewriter
        .replaceOpWithNewOp<mlir::LLVM::LoadOp>(
            op, typeConverter.convertType(op.getType()), address)
        .setTbaaAttr(codeGenState.getTBAAAccess(pylir::TbaaAccessType::Handle));
    return mlir::success();
  }
};

struct StoreOpConversion : public ConvertPylirOpToLLVMPattern<Py::StoreOp> {
  using ConvertPylirOpToLLVMPattern<Py::StoreOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), rewriter.getType<mlir::LLVM::LLVMPointerType>(),
        adaptor.getGlobalAttr());
    rewriter
        .replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(),
                                                 address)
        .setTbaaAttr(codeGenState.getTBAAAccess(pylir::TbaaAccessType::Handle));
    return mlir::success();
  }
};

struct IsOpConversion : public ConvertPylirOpToLLVMPattern<Py::IsOp> {
  using ConvertPylirOpToLLVMPattern<Py::IsOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::IsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::eq, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

struct IsUnboundValueOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::IsUnboundValueOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::IsUnboundValueOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::IsUnboundValueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto null = rewriter.create<mlir::LLVM::ZeroOp>(
        op.getLoc(), adaptor.getValue().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::eq, adaptor.getValue(), null);
    return mlir::success();
  }
};

struct TypeOfOpConversion : public ConvertPylirOpToLLVMPattern<Py::TypeOfOp> {
  using ConvertPylirOpToLLVMPattern<Py::TypeOfOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TypeOfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto model = pyObjectModel(rewriter, adaptor.getObject());
    rewriter.replaceOp(op, model.typePtr(op.getLoc()).load(op.getLoc()));
    return mlir::success();
  }
};

struct TupleGetItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::TupleGetItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::TupleGetItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TupleGetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyTupleModel(rewriter, adaptor.getTuple())
                               .trailingArray(op.getLoc())
                               .at(op.getLoc(), adaptor.getIndex())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct DictLenOpConversion : public ConvertPylirOpToLLVMPattern<Py::DictLenOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::DictLenOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto sequence = pyDictModel(rewriter, adaptor.getInput());
    rewriter.replaceOp(
        op,
        sequence.bufferPtr(op.getLoc()).size(op.getLoc()).load(op.getLoc()));
    return mlir::success();
  }
};

struct TupleLenOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::TupleLenOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::TupleLenOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TupleLenOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyTupleModel(rewriter, adaptor.getInput())
                               .size(op.getLoc())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct TupleContainsOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::TupleContainsOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TupleContainsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto* block = op->getBlock();
    auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
    endBlock->addArgument(rewriter.getI1Type(), op.getLoc());
    rewriter.setInsertionPointToEnd(block);

    auto tupleModel = pyTupleModel(rewriter, adaptor.getTuple());
    auto size = tupleModel.size(op.getLoc()).load(op.getLoc());
    auto zero =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 0);

    auto* conditionBlock = new mlir::Block;
    conditionBlock->addArgument(getIndexType(), op.getLoc());
    rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), zero, conditionBlock);

    conditionBlock->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(conditionBlock);
    auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::ne,
        conditionBlock->getArgument(0), size);
    auto* body = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), cmp, body, endBlock,
                                          mlir::ValueRange{cmp});

    body->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(body);
    auto element = tupleModel.trailingArray(op.getLoc())
                       .at(op.getLoc(), conditionBlock->getArgument(0))
                       .load(op.getLoc());
    auto isElement = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::eq, element,
        adaptor.getElement());
    auto one =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 1);
    mlir::Value incremented = rewriter.create<mlir::LLVM::AddOp>(
        op.getLoc(), conditionBlock->getArgument(0), one);
    mlir::Value trueV = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(true));
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isElement, endBlock,
                                          trueV, conditionBlock, incremented);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.replaceOp(op, endBlock->getArgument(0));
    return mlir::success();
  }
};

struct ListLenOpConversion : public ConvertPylirOpToLLVMPattern<Py::ListLenOp> {
  using ConvertPylirOpToLLVMPattern<Py::ListLenOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ListLenOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyListModel(rewriter, adaptor.getList())
                               .size(op.getLoc())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct ListGetItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ListGetItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ListGetItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ListGetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyListModel(rewriter, adaptor.getList())
                               .tuplePtr(op.getLoc())
                               .load(op.getLoc())
                               .trailingArray(op.getLoc())
                               .at(op.getLoc(), adaptor.getIndex())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct ListSetItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ListSetItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ListSetItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ListSetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    pyListModel(rewriter, adaptor.getList())
        .tuplePtr(op.getLoc())
        .load(op.getLoc())
        .trailingArray(op.getLoc())
        .at(op.getLoc(), adaptor.getIndex())
        .store(op.getLoc(), adaptor.getElement());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ListResizeOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ListResizeOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ListResizeOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ListResizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto* block = op->getBlock();
    auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
    rewriter.setInsertionPointToEnd(block);

    auto list = pyListModel(rewriter, adaptor.getList());
    auto tuplePtr = list.tuplePtr(op.getLoc()).load(op.getLoc());
    auto sizePtr = list.size(op.getLoc());
    auto size = sizePtr.load(op.getLoc());
    auto oneIndex =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 1);

    auto capacityPtr = tuplePtr.size(op.getLoc());
    auto capacity = capacityPtr.load(op.getLoc());
    auto notEnoughCapacity = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::ult, capacity,
        adaptor.getLength());
    auto* growBlock = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), notEnoughCapacity,
                                          growBlock, endBlock);

    growBlock->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(growBlock);
    {
      mlir::Value newCapacity =
          rewriter.create<mlir::LLVM::ShlOp>(op.getLoc(), capacity, oneIndex);
      newCapacity = rewriter.create<mlir::LLVM::UMaxOp>(
          op.getLoc(), newCapacity, adaptor.getLength());

      auto tupleType = rewriter.create<Py::ConstantOp>(
          op.getLoc(),
          Py::GlobalValueAttr::get(getContext(), Builtins::Tuple.name));
      mlir::Value tupleMemory = rewriter.create<Mem::GCAllocObjectOp>(
          op.getLoc(), tupleType, newCapacity);
      tupleMemory = unrealizedConversion(rewriter, tupleMemory, typeConverter);

      auto newTupleModel = pyTupleModel(rewriter, tupleMemory);
      newTupleModel.size(op.getLoc()).store(op.getLoc(), newCapacity);
      auto trailingArray = newTupleModel.trailingArray(op.getLoc());
      auto array = trailingArray.at(op.getLoc(), 0);
      auto prevArray = tuplePtr.trailingArray(op.getLoc()).at(op.getLoc(), 0);
      auto elementTypeSize =
          createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                  typeConverter.getPointerBitwidth() / 8);
      auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size,
                                                        elementTypeSize);
      rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), array, prevArray,
                                            inBytes, /*isVolatile=*/false);
      list.tuplePtr(op.getLoc()).store(op.getLoc(), newTupleModel);
    }
    rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{},
                                      endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    sizePtr.store(op.getLoc(), adaptor.getLength());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct DictTryGetItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::DictTryGetItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::DictTryGetItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::DictTryGetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto dict = pyDictModel(rewriter, adaptor.getDict());
    auto result = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_dict_lookup,
        {dict, adaptor.getKey(), adaptor.getHash()});
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct DictSetItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::DictSetItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::DictSetItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::DictSetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto dict = pyDictModel(rewriter, adaptor.getDict());
    codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_dict_insert,
        {dict, adaptor.getKey(), adaptor.getHash(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct DictDelItemOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::DictDelItemOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::DictDelItemOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::DictDelItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto dict = pyDictModel(rewriter, adaptor.getDict());
    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::pylir_dict_erase,
                                   {dict, adaptor.getKey(), adaptor.getHash()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct IntToIndexOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::IntToIndexOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::IntToIndexOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::IntToIndexOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto call = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::mp_get_i64,
        pyIntModel(rewriter, adaptor.getInput()).mpInt(op.getLoc()));
    if (call.getType() != typeConverter.convertType(op.getType()))
      call = rewriter.create<mlir::LLVM::TruncOp>(
          op.getLoc(), typeConverter.convertType(op.getType()), call);

    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct InitIntAddOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitIntAddOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitIntAddOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitIntAddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto memoryInt =
        pyIntModel(rewriter, adaptor.getMemory()).mpInt(op.getLoc());
    auto lhsInt = pyIntModel(rewriter, adaptor.getLhs()).mpInt(op.getLoc());
    auto rhsInt = pyIntModel(rewriter, adaptor.getRhs()).mpInt(op.getLoc());

    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::mp_init, memoryInt);
    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::mp_add,
                                   {lhsInt, rhsInt, memoryInt});

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct IntCmpOpConversion : public ConvertPylirOpToLLVMPattern<Py::IntCmpOp> {
  using ConvertPylirOpToLLVMPattern<Py::IntCmpOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::IntCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto lhsInt = pyIntModel(rewriter, adaptor.getLhs()).mpInt(op.getLoc());
    auto rhsInt = pyIntModel(rewriter, adaptor.getRhs()).mpInt(op.getLoc());
    auto result = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::mp_cmp, {lhsInt, rhsInt});
    mp_ord mpOrd;
    mlir::LLVM::ICmpPredicate predicate;
    switch (adaptor.getPred()) {
    case Py::IntCmpKind::eq:
      mpOrd = MP_EQ;
      predicate = mlir::LLVM::ICmpPredicate::eq;
      break;
    case Py::IntCmpKind::ne:
      mpOrd = MP_EQ;
      predicate = mlir::LLVM::ICmpPredicate::ne;
      break;
    case Py::IntCmpKind::lt:
      mpOrd = MP_LT;
      predicate = mlir::LLVM::ICmpPredicate::eq;
      break;
    case Py::IntCmpKind::le:
      mpOrd = MP_GT;
      predicate = mlir::LLVM::ICmpPredicate::ne;
      break;
    case Py::IntCmpKind::gt:
      mpOrd = MP_GT;
      predicate = mlir::LLVM::ICmpPredicate::eq;
      break;
    case Py::IntCmpKind::ge:
      mpOrd = MP_LT;
      predicate = mlir::LLVM::ICmpPredicate::ne;
      break;
    default: PYLIR_UNREACHABLE;
    }

    mlir::Type intType = typeConverter.getPlatformABI().getInt(getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, predicate, result,
        rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), mlir::IntegerAttr::get(intType, mpOrd)));
    return mlir::success();
  }
};

struct BoolToI1OpConversion
    : public ConvertPylirOpToLLVMPattern<Py::BoolToI1Op> {
  using ConvertPylirOpToLLVMPattern<
      Py::BoolToI1Op>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::BoolToI1Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto load = pyIntModel(rewriter, adaptor.getInput())
                    .mpInt(op.getLoc())
                    .used(op.getLoc())
                    .load(op.getLoc());
    auto zeroI = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), load.getType(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::ne, load, zeroI);
    return mlir::success();
  }
};

struct ObjectHashOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ObjectHashOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ObjectHashOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ObjectHashOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    // TODO: proper hash
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(
        op, typeConverter.convertType(op.getType()), adaptor.getObject());
    return mlir::success();
  }
};

struct ObjectIdOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::ObjectIdOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::ObjectIdOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ObjectIdOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(
        op, typeConverter.convertType(op.getType()), adaptor.getObject());
    return mlir::success();
  }
};

struct TypeMROOpConversion : public ConvertPylirOpToLLVMPattern<Py::TypeMROOp> {
  using ConvertPylirOpToLLVMPattern<Py::TypeMROOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TypeMROOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyTypeModel(rewriter, adaptor.getTypeObject())
                               .mroPtr(op.getLoc())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct TypeSlotsOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::TypeSlotsOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::TypeSlotsOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::TypeSlotsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyTypeModel(rewriter, adaptor.getTypeObject())
                               .instanceSlotsPtr(op.getLoc())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct StrEqualOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::StrEqualOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::StrEqualOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::StrEqualOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto* block = op->getBlock();
    auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
    endBlock->addArgument(rewriter.getI1Type(), op.getLoc());
    rewriter.setInsertionPointToEnd(block);

    auto sameObject = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::eq, adaptor.getLhs(),
        adaptor.getRhs());
    auto* isNot = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sameObject, endBlock,
                                          mlir::ValueRange{sameObject}, isNot,
                                          mlir::ValueRange{});

    isNot->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(isNot);
    auto lhs = pyStringModel(rewriter, adaptor.getLhs()).buffer(op.getLoc());
    auto rhs = pyStringModel(rewriter, adaptor.getRhs()).buffer(op.getLoc());
    auto lhsLen = lhs.size(op.getLoc()).load(op.getLoc());
    auto rhsLen = rhs.size(op.getLoc()).load(op.getLoc());
    auto sizeEqual = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, rhsLen);
    auto* sizeEqualBlock = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeEqual,
                                          sizeEqualBlock, endBlock,
                                          mlir::ValueRange{sizeEqual});

    sizeEqualBlock->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(sizeEqualBlock);
    auto zeroI =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 0);
    auto sizeZero = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, zeroI);
    auto* bufferCmp = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeZero, endBlock,
                                          mlir::ValueRange{sizeZero}, bufferCmp,
                                          mlir::ValueRange{});

    bufferCmp->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(bufferCmp);
    auto lhsBuffer = lhs.elementPtr(op.getLoc()).load(op.getLoc());
    auto rhsBuffer = rhs.elementPtr(op.getLoc()).load(op.getLoc());
    auto result = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::memcmp,
        {lhsBuffer, rhsBuffer, lhsLen});

    zeroI = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), typeConverter.getPlatformABI().getInt(getContext()),
        rewriter.getI32IntegerAttr(0));
    auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::eq, result, zeroI);
    rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{isZero},
                                      endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.replaceOp(op, {endBlock->getArgument(0)});
    return mlir::success();
  }
};

struct StrHashOpConversion : public ConvertPylirOpToLLVMPattern<Py::StrHashOp> {
  using ConvertPylirOpToLLVMPattern<Py::StrHashOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::StrHashOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto str = pyStringModel(rewriter, adaptor.getObject());
    auto hash = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_str_hash, str);
    rewriter.replaceOp(op, hash);
    return mlir::success();
  }
};

struct PrintOpConversion : public ConvertPylirOpToLLVMPattern<Py::PrintOp> {
  using ConvertPylirOpToLLVMPattern<Py::PrintOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto str = pyStringModel(rewriter, adaptor.getString());
    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::pylir_print, str);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GetSlotOpConversion : public ConvertPylirOpToLLVMPattern<Py::GetSlotOp> {
  using ConvertPylirOpToLLVMPattern<Py::GetSlotOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::GetSlotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto typeObj = pyTypeModel(rewriter, adaptor.getObject())
                       .typePtr(op.getLoc())
                       .load(op.getLoc());
    auto offset = typeObj.offset(op.getLoc()).load(op.getLoc());
    mlir::Value index = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset,
                                                           adaptor.getSlot());

    auto ptrType = adaptor.getObject().getType();
    auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), ptrType, ptrType,
                                                  adaptor.getObject(), index);
    auto slot = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(),
                                                    gep.getElemType(), gep);
    slot.setTbaaAttr(codeGenState.getTBAAAccess(TbaaAccessType::Slots));

    rewriter.replaceOp(op, mlir::Value(slot));
    return mlir::success();
  }
};

struct SetSlotOpConversion : public ConvertPylirOpToLLVMPattern<Py::SetSlotOp> {
  using ConvertPylirOpToLLVMPattern<Py::SetSlotOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::SetSlotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto typeObj = pyTypeModel(rewriter, adaptor.getObject())
                       .typePtr(op.getLoc())
                       .load(op.getLoc());
    auto offset = typeObj.offset(op.getLoc()).load(op.getLoc());
    mlir::Value index = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset,
                                                           adaptor.getSlot());

    auto ptrType = adaptor.getObject().getType();
    auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), ptrType, ptrType,
                                                  adaptor.getObject(), index);

    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), gep)
        .setTbaaAttr(codeGenState.getTBAAAccess(TbaaAccessType::Slots));
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RaiseOpConversion : public ConvertPylirOpToLLVMPattern<Py::RaiseOp> {
  using ConvertPylirOpToLLVMPattern<Py::RaiseOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::RaiseOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::pylir_raise,
                                   {adaptor.getException()});
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

struct RaiseExOpConversion : public ConvertPylirOpToLLVMPattern<Py::RaiseExOp> {
  using ConvertPylirOpToLLVMPattern<Py::RaiseExOp>::ConvertPylirOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Py::RaiseExOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> arguments = {adaptor.getException()};
    llvm::append_range(arguments, adaptor.getUnwindDestOperands());
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, arguments,
                                            op.getExceptionPath());
    return mlir::success();
  }
};

struct CallOpConversion : public ConvertPylirOpToLLVMPattern<Py::CallOp> {
  using ConvertPylirOpToLLVMPattern<Py::CallOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTypes;
    [[maybe_unused]] auto result =
        typeConverter.convertTypes(op->getResultTypes(), resultTypes);
    PYLIR_ASSERT(mlir::succeeded(result));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, resultTypes, adaptor.getCalleeAttr(), adaptor.getCallOperands());
    return mlir::success();
  }
};

struct FunctionCallOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::FunctionCallOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::FunctionCallOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::FunctionCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTypes;
    [[maybe_unused]] auto result =
        typeConverter.convertTypes(op->getResultTypes(), resultTypes);
    PYLIR_ASSERT(mlir::succeeded(result));
    auto callee = pyFunctionModel(rewriter, adaptor.getFunction())
                      .funcPtr(op.getLoc())
                      .load(op.getLoc());
    llvm::SmallVector<mlir::Value> operands{callee};
    operands.append(adaptor.getCallOperands().begin(),
                    adaptor.getCallOperands().end());
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, resultTypes, operands);

    return mlir::success();
  }
};

struct FunctionGetClosureArgOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::FunctionGetClosureArgOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::FunctionGetClosureArgOp>::ConvertPylirOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Py::FunctionGetClosureArgOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> convertedTypes;
    if (failed(typeConverter.convertTypes(
            llvm::to_vector(
                adaptor.getClosureTypes().getAsValueRange<mlir::TypeAttr>()),
            convertedTypes)))
      return failure();

    rewriter.replaceOp(
        op, pyFunctionModel(rewriter, adaptor.getFunction(), convertedTypes)
                .closureArgument(op.getLoc(), adaptor.getIndex())
                .load(op.getLoc()));
    return success();
  }
};

template <class T>
struct InvokeOpsConversion : public ConvertPylirOpToLLVMPattern<T> {
  using ConvertPylirOpToLLVMPattern<T>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTypes;
    [[maybe_unused]] auto result =
        this->typeConverter.convertTypes(op->getResultTypes(), resultTypes);
    PYLIR_ASSERT(mlir::succeeded(result));
    auto ip = rewriter.saveInsertionPoint();
    auto* endBlock = rewriter.createBlock(
        op->getParentRegion(),
        mlir::Region::iterator{op->getBlock()->getNextNode()});
    rewriter.restoreInsertionPoint(ip);
    if constexpr (std::is_same_v<T, Py::InvokeOp>) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
          op, resultTypes, adaptor.getCalleeAttr(), adaptor.getCallOperands(),
          op.getHappyPath(), adaptor.getNormalDestOperands(), endBlock,
          mlir::ValueRange{});
    } else {
      auto callee = this->pyFunctionModel(rewriter, adaptor.getFunction())
                        .funcPtr(op.getLoc())
                        .load(op.getLoc());
      llvm::SmallVector<mlir::Value> operands{callee};
      operands.append(adaptor.getCallOperands().begin(),
                      adaptor.getCallOperands().end());
      assert(resultTypes.size() == 1 &&
             "Lowering currently only supports one return type");
      rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
          op,
          LLVM::LLVMFunctionType::get(
              resultTypes.front(),
              llvm::to_vector(adaptor.getCallOperands().getTypes())),
          /*callee=*/nullptr, operands, op.getHappyPath(),
          adaptor.getNormalDestOperands(), endBlock, mlir::ValueRange{});
    }

    rewriter.setInsertionPointToStart(endBlock);
    mlir::Value catchType;
    {
      mlir::OpBuilder::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
      catchType = this->codeGenState.getConstant(
          op.getLoc(), rewriter,
          Py::GlobalValueAttr::get(this->getContext(),
                                   Builtins::BaseException.name));
    }
    // We use a integer of pointer width instead of a pointer to keep it opaque
    // to statepoint passes. Those do not support aggregates in aggregates.
    auto literal = mlir::LLVM::LLVMStructType::getLiteral(
        this->getContext(),
        {this->getIntPtrType(REF_ADDRESS_SPACE), rewriter.getI32Type()});
    auto landingPad = rewriter.create<mlir::LLVM::LandingpadOp>(
        op.getLoc(), literal, catchType);
    mlir::Value exceptionHeader =
        rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), landingPad, 0);
    {
      // Itanium ABI mandates a pointer to the exception header be returned by
      // the landing pad. So we need to subtract the offset of the exception
      // header inside of PyBaseException to get to it.
      auto pyBaseException = this->typeConverter.getPyBaseExceptionType();
      auto unwindHeader = this->typeConverter.getUnwindHeaderType();
      std::size_t offsetOf = 0;
      for (const auto& iter : pyBaseException.getBody()) {
        offsetOf = llvm::alignTo(
            offsetOf, this->typeConverter.getPlatformABI().getAlignOf(iter));
        if (iter == unwindHeader)
          break;

        offsetOf += this->typeConverter.getPlatformABI().getSizeOf(iter);
      }
      auto byteOffset = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), exceptionHeader.getType(),
          rewriter.getI64IntegerAttr(offsetOf));
      exceptionHeader = rewriter.create<mlir::LLVM::SubOp>(
          op.getLoc(), exceptionHeader, byteOffset);
    }
    auto exceptionObject = rewriter.create<mlir::LLVM::IntToPtrOp>(
        op.getLoc(),
        rewriter.getType<mlir::LLVM::LLVMPointerType>(REF_ADDRESS_SPACE),
        exceptionHeader);
    auto ops = llvm::to_vector(op.getUnwindDestOperands());
    ops.insert(ops.begin(), exceptionObject);
    rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), ops, op.getExceptionPath());
    return mlir::success();
  }
};

struct GCAllocObjectConstTypeConversion
    : public ConvertPylirOpToLLVMPattern<Mem::GCAllocObjectOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto constant = op.getTypeObject().getDefiningOp<Py::ConstantOp>();
    if (!constant)
      return mlir::failure();

    auto typeAttr = dyn_cast<TypeAttrInterface>(constant.getConstant());
    if (!typeAttr)
      return mlir::failure();

    // I could create GEP here to read the offset component of the type object,
    // but LLVM is not aware that the size component is const, even if the rest
    // of the type isn't. So instead we calculate the size here again to have it
    // be a constant.
    auto layoutType = typeConverter.getLayoutType(constant.getConstant());
    if (!layoutType)
      return mlir::failure();

    mlir::Type instanceType = typeConverter.mapLayoutTypeToLLVM(*layoutType);
    auto instanceSize = createIndexAttrConstant(
        rewriter, op.getLoc(), getIndexType(),
        typeConverter.getPlatformABI().getSizeOf(instanceType));
    auto pointerSize =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    auto slotSize = rewriter.create<mlir::LLVM::MulOp>(
        op.getLoc(), adaptor.getTrailingItems(), pointerSize);
    auto inBytes =
        rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), slotSize, instanceSize);
    auto memory = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_gc_alloc,
        {inBytes});
    auto zeroI8 = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
    rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes,
                                          /*isVolatile=*/false);
    pyObjectModel(rewriter, memory)
        .typePtr(op.getLoc())
        .store(op.getLoc(), adaptor.getTypeObject());
    rewriter.replaceOp(op, memory);
    return mlir::success();
  }
};

struct GCAllocObjectOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::GCAllocObjectOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto len = adaptor.getTrailingItems();
    auto offset = pyTypeModel(rewriter, adaptor.getTypeObject())
                      .offset(op.getLoc())
                      .load(op.getLoc());
    auto size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, len);
    auto pointerSize =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    auto inBytes =
        rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, pointerSize);
    auto memory = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_gc_alloc,
        {inBytes});
    auto zeroI8 = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
    rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes,
                                          /*isVolatile=*/false);
    pyObjectModel(rewriter, memory)
        .typePtr(op.getLoc())
        .store(op.getLoc(), adaptor.getTypeObject());
    rewriter.replaceOp(op, memory);
    return mlir::success();
  }
};

struct StackAllocObjectOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::StackAllocObjectOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::StackAllocObjectOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::StackAllocObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type elementType;
    mlir::Value memory;
    {
      mlir::OpBuilder::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToStart(
          &op->getParentRegion()->getBlocks().front());

      elementType = typeConverter.mapLayoutTypeToLLVM(
          adaptor.getLayout(), adaptor.getTrailingItems().getZExtValue());
      auto one = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      memory = rewriter.create<mlir::LLVM::AllocaOp>(
          op.getLoc(),
          rewriter.getType<mlir::LLVM::LLVMPointerType>(REF_ADDRESS_SPACE),
          elementType, one);
    }

    std::size_t elementTypeSize =
        typeConverter.getPlatformABI().getSizeOf(elementType);
    rewriter.create<mlir::LLVM::LifetimeStartOp>(op.getLoc(), elementTypeSize,
                                                 memory);

    auto inBytes = createIndexAttrConstant(rewriter, op.getLoc(),
                                           getIndexType(), elementTypeSize);
    auto zeroI8 = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
    rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes,
                                          /*isVolatile=*/false);
    pyObjectModel(rewriter, memory)
        .typePtr(op.getLoc())
        .store(op.getLoc(), adaptor.getTypeObject());
    rewriter.replaceOp(op, memory);
    return mlir::success();
  }
};

struct GCAllocFunctionOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::GCAllocFunctionOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::GCAllocFunctionOp>::ConvertPylirOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mem::GCAllocFunctionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> types;
    if (failed(typeConverter.convertTypes(
            llvm::to_vector(adaptor.getClosureArgsTypes()
                                .getAsValueRange<mlir::TypeAttr>()),
            types)))
      return failure();

    LLVM::LLVMStructType type = typeConverter.getPyFunctionType(
        std::initializer_list<int>{
#define FUNCTION_SLOT(...) 0,
#include <pylir/Interfaces/Slots.def>
        }
            .size(),
        types);
    auto pointerType = rewriter.getType<LLVM::LLVMPointerType>();
    Value null = rewriter.create<LLVM::ZeroOp>(op.getLoc(), pointerType);
    Value typeSize = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), pointerType, type, null, ArrayRef<LLVM::GEPArg>{1});
    typeSize = rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), getIndexType(),
                                                 typeSize);

    Value memory = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::pylir_gc_alloc,
        {typeSize});
    Value zeroI8 = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
    rewriter.create<LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, typeSize,
                                    /*isVolatile=*/false);
    pyObjectModel(rewriter, memory)
        .typePtr(op.getLoc())
        .store(op.getLoc(),
               codeGenState.getConstant(op.getLoc(), rewriter,
                                        rewriter.getType<Py::GlobalValueAttr>(
                                            Builtins::Function.name)));
    rewriter.replaceOp(op, memory);
    return mlir::success();
  }
};

struct InitObjectOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitObjectOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitObjectOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTupleOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTupleOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitTupleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto tuple = pyTupleModel(rewriter, adaptor.getMemory());
    auto size = createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                        adaptor.getInitializer().size());

    tuple.size(op.getLoc()).store(op.getLoc(), size);
    auto trailing = tuple.trailingArray(op.getLoc());
    for (const auto& iter : llvm::enumerate(adaptor.getInitializer()))
      trailing.at(op.getLoc(), iter.index()).store(op.getLoc(), iter.value());

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitListOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitListOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitListOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto list = pyListModel(rewriter, adaptor.getMemory());
    auto size = createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                        adaptor.getInitializer().size());

    list.size(op.getLoc()).store(op.getLoc(), size);

    auto tupleType = rewriter.create<Py::ConstantOp>(
        op.getLoc(),
        Py::GlobalValueAttr::get(getContext(), Builtins::Tuple.name));
    auto tupleMemory =
        rewriter.create<Mem::GCAllocObjectOp>(op.getLoc(), tupleType, size);
    mlir::Value tupleInit = rewriter.create<Mem::InitTupleOp>(
        op.getLoc(), op.getType(), tupleMemory, adaptor.getInitializer());
    auto tuplePtr = list.tuplePtr(op.getLoc());
    tuplePtr.store(op.getLoc(),
                   unrealizedConversion(rewriter, tupleInit, typeConverter));

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTupleFromListOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTupleFromListOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitTupleFromListOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitTupleFromListOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto tuple = pyTupleModel(rewriter, adaptor.getMemory());
    auto list = pyListModel(rewriter, adaptor.getInitializer());

    auto size = list.size(op.getLoc()).load(op.getLoc());
    tuple.size(op.getLoc()).store(op.getLoc(), size);

    auto sizeOf =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    auto inBytes =
        rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

    auto array = tuple.trailingArray(op.getLoc()).at(op.getLoc(), 0);
    auto listArray = list.tuplePtr(op.getLoc())
                         .load(op.getLoc())
                         .trailingArray(op.getLoc())
                         .at(op.getLoc(), 0);
    rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), array, listArray,
                                          inBytes, /*isVolatile=*/false);

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTupleCopyOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTupleCopyOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitTupleCopyOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitTupleCopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto destTuple = pyTupleModel(rewriter, adaptor.getMemory());
    auto sourceTuple = pyTupleModel(rewriter, adaptor.getInitializer());

    auto size = sourceTuple.size(op.getLoc()).load(op.getLoc());
    destTuple.size(op.getLoc()).store(op.getLoc(), size);

    auto sizeOf =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    auto inBytes =
        rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

    auto array = destTuple.trailingArray(op.getLoc()).at(op.getLoc(), 0);
    auto listArray = sourceTuple.trailingArray(op.getLoc()).at(op.getLoc(), 0);
    rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), array, listArray,
                                          inBytes, /*isVolatile=*/false);

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTupleDropFrontOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTupleDropFrontOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitTupleDropFrontOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitTupleDropFrontOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto tuple = pyTupleModel(rewriter, adaptor.getMemory());
    auto prevTuple = pyTupleModel(rewriter, adaptor.getTuple());

    mlir::Value size = prevTuple.size(op.getLoc()).load(op.getLoc());
    size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size,
                                              adaptor.getCount());

    tuple.size(op.getLoc()).store(op.getLoc(), size);

    auto sizeOf =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    auto inBytes =
        rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

    auto array = tuple.trailingArray(op.getLoc()).at(op.getLoc(), 0);
    auto prevArray = prevTuple.trailingArray(op.getLoc())
                         .at(op.getLoc(), adaptor.getCount());
    rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), array, prevArray,
                                          inBytes, /*isVolatile=*/false);
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTuplePrependOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTuplePrependOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitTuplePrependOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitTuplePrependOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto tuple = pyTupleModel(rewriter, adaptor.getMemory());
    auto prevTuple = pyTupleModel(rewriter, adaptor.getTuple());

    mlir::Value size = prevTuple.size(op.getLoc()).load(op.getLoc());
    auto oneI =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 1);
    auto resultSize =
        rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, oneI);

    tuple.size(op.getLoc()).store(op.getLoc(), resultSize);
    tuple.trailingArray(op.getLoc())
        .at(op.getLoc(), 0)
        .store(op.getLoc(), adaptor.getElement());

    auto sizeOf =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(),
                                typeConverter.getPointerBitwidth() / 8);
    mlir::Value inBytes =
        rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

    auto array = tuple.trailingArray(op.getLoc()).at(op.getLoc(), 1);
    auto prevArray = prevTuple.trailingArray(op.getLoc()).at(op.getLoc(), 0);
    rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), array, prevArray,
                                          inBytes, /*isVolatile=*/false);
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitIntUnsignedOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitIntUnsignedOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitIntUnsignedOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitIntUnsignedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto mpIntPointer =
        pyIntModel(rewriter, adaptor.getMemory()).mpInt(op.getLoc());
    auto value = adaptor.getInitializer();
    if (value.getType() != rewriter.getI64Type())
      value = rewriter.create<mlir::LLVM::ZExtOp>(op.getLoc(),
                                                  rewriter.getI64Type(), value);

    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::mp_init_u64,
                                   {mpIntPointer, value});
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitIntSignedOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitIntSignedOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitIntSignedOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitIntSignedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto mpIntPointer =
        pyIntModel(rewriter, adaptor.getMemory()).mpInt(op.getLoc());
    auto value = adaptor.getInitializer();
    if (value.getType() != rewriter.getI64Type())
      value = rewriter.create<mlir::LLVM::ZExtOp>(op.getLoc(),
                                                  rewriter.getI64Type(), value);

    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::mp_init_i64,
                                   {mpIntPointer, value});
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitStrOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitStrOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitStrOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitStrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto string =
        pyStringModel(rewriter, adaptor.getMemory()).buffer(op.getLoc());

    mlir::Value size =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 0);
    for (auto iter : adaptor.getStrings()) {
      auto sizeLoaded = pyStringModel(rewriter, iter)
                            .buffer(op.getLoc())
                            .size(op.getLoc())
                            .load(op.getLoc());
      size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
    }

    string.size(op.getLoc()).store(op.getLoc(), size);
    string.capacity(op.getLoc()).store(op.getLoc(), size);

    auto array = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::malloc, {size});
    string.elementPtr(op.getLoc()).store(op.getLoc(), array);

    size = createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 0);
    for (auto iter : adaptor.getStrings()) {
      auto iterString = pyStringModel(rewriter, iter).buffer(op.getLoc());
      auto sizeLoaded = iterString.size(op.getLoc()).load(op.getLoc());
      auto sourceLoaded = iterString.elementPtr(op.getLoc()).load(op.getLoc());
      auto dest = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), array.getType(), rewriter.getI8Type(), array, size);
      rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), dest, sourceLoaded,
                                            sizeLoaded, /*isVolatile=*/false);
      size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
    }
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitStrFromIntOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitStrFromIntOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitStrFromIntOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitStrFromIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto string =
        pyStringModel(rewriter, adaptor.getMemory()).buffer(op.getLoc());
    auto mpIntPtr =
        pyIntModel(rewriter, adaptor.getInteger()).mpInt(op.getLoc());
    auto sizePtr = string.size(op.getLoc());
    auto ten = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), typeConverter.getPlatformABI().getInt(getContext()),
        rewriter.getI32IntegerAttr(10));
    codeGenState.createRuntimeCall(
        op.getLoc(), rewriter,
        CodeGenState::Runtime::mp_radix_size_overestimate,
        {mpIntPtr, ten, sizePtr});
    auto capacity = sizePtr.load(op.getLoc());
    auto array = codeGenState.createRuntimeCall(
        op.getLoc(), rewriter, CodeGenState::Runtime::malloc, {capacity});
    codeGenState.createRuntimeCall(op.getLoc(), rewriter,
                                   CodeGenState::Runtime::mp_to_radix,
                                   {mpIntPtr, array, capacity, sizePtr, ten});

    // mp_to_radix sadly includes the NULL terminator that it uses in size...
    mlir::Value size = sizePtr.load(op.getLoc());
    auto oneI =
        createIndexAttrConstant(rewriter, op.getLoc(), getIndexType(), 1);
    size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size, oneI);
    sizePtr.store(op.getLoc(), size);

    string.capacity(op.getLoc()).store(op.getLoc(), capacity);
    string.elementPtr(op.getLoc()).store(op.getLoc(), array);

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitFuncOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitFuncOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitFuncOp>::ConvertPylirOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mem::InitFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Value address = rewriter.create<LLVM::AddressOfOp>(
        op.getLoc(), rewriter.getType<LLVM::LLVMPointerType>(),
        adaptor.getInitializer());

    PyFunctionModel model = pyFunctionModel(
        rewriter, adaptor.getMemory(), adaptor.getClosureArgs().getTypes());

    model.funcPtr(op.getLoc()).store(op.getLoc(), address);

    // Store the closure's size.
    unsigned byteCount =
        typeConverter.getClosureArgsBytes(adaptor.getClosureArgs().getTypes());
    Value byteCountC = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(byteCount));
    model.closureSizePtr(op.getLoc()).store(op.getLoc(), byteCountC);

    for (auto [index, value] : llvm::enumerate(adaptor.getClosureArgs()))
      model.closureArgument(op.getLoc(), index).store(op.getLoc(), value);

    // Note where references live within the closure by populating the 'refMask'
    // bitset.
    unsigned pointerBitwidth = typeConverter.getPointerBitwidth();
    SmallVector<std::uint8_t> refMask(
        llvm::divideCeil(byteCount, pointerBitwidth));
    unsigned currentOffset = 0;
    PlatformABI& platformAbi = typeConverter.getPlatformABI();
    for (auto [newType, oldType] :
         llvm::zip_equal(adaptor.getClosureArgs().getTypes(),
                         op.getClosureArgs().getTypes())) {
      currentOffset =
          llvm::alignTo(currentOffset, platformAbi.getAlignOf(newType));
      if (isa<DynamicType>(oldType)) {
        unsigned index = currentOffset / (pointerBitwidth / 8);
        refMask[index / 8] |= 1 << (index % 8);
      }
      currentOffset += platformAbi.getSizeOf(newType);
    }

    auto array = model.refInClosureBitfield(op.getLoc(),
                                            adaptor.getClosureArgs().size());
    for (auto [index, value] : llvm::enumerate(refMask)) {
      Value mask = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI8IntegerAttr(value));
      array.at(op.getLoc(), index).store(op.getLoc(), mask);
    }

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitDictOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitDictOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitDictOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitDictOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitFloatOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitFloatOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitFloatOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Mem::InitFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    pyFloatModel(rewriter, adaptor.getMemory())
        .doubleValue(op.getLoc())
        .store(op.getLoc(), adaptor.getInitializer());
    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct InitTypeOpConversion
    : public ConvertPylirOpToLLVMPattern<Mem::InitTypeOp> {
  using ConvertPylirOpToLLVMPattern<
      Mem::InitTypeOp>::ConvertPylirOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mem::InitTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    PyTypeModel model = pyTypeModel(rewriter, adaptor.getMemory());
    Value typeType = rewriter.create<Py::ConstantOp>(
        op.getLoc(),
        rewriter.getAttr<Py::GlobalValueAttr>(Builtins::Type.name));
    typeType = unrealizedConversion(rewriter, typeType, typeConverter);
    model.typePtr(op.getLoc()).store(op.getLoc(), typeType);

    Value mroTuple = adaptor.getMroTuple();
    mroTuple =
        unrealizedConversion(rewriter, mroTuple, op.getMroTuple().getType());
    Value element = unrealizedConversion(rewriter, adaptor.getMemory(),
                                         op.getResult().getType());
    Value mroTupleMemory = unrealizedConversion(
        rewriter, adaptor.getMroTupleMemory(), op.getMemory().getType());
    mroTuple = rewriter.create<Mem::InitTuplePrependOp>(
        op.getLoc(), mroTupleMemory, element, mroTuple);
    mroTuple = unrealizedConversion(rewriter, mroTuple, typeConverter);
    model.mroPtr(op.getLoc()).store(op.getLoc(), mroTuple);

    model.instanceSlotsPtr(op.getLoc())
        .store(op.getLoc(), adaptor.getSlotsTuple());

    // TODO: Layout and offset need to be computed at some point.
    //       They seem related, is the offset redundant?
    Value objectType = rewriter.create<Py::ConstantOp>(
        op.getLoc(),
        rewriter.getAttr<Py::GlobalValueAttr>(Builtins::Object.name));
    Value layout = unrealizedConversion(rewriter, objectType, typeConverter);
    model.layoutPtr(op.getLoc()).store(op.getLoc(), layout);
    {
      Type instanceType = typeConverter.getPyObjectType();
      auto asCount = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), typeConverter.getIndexType(),
          rewriter.getI32IntegerAttr(
              typeConverter.getPlatformABI().getSizeOf(instanceType) /
              (typeConverter.getPointerBitwidth() / 8)));
      model.offset(op.getLoc()).store(op.getLoc(), asCount);
    }
    model.slotsArray(op.getLoc())
        .at(op.getLoc(), Builtins::TypeSlots::Name)
        .store(op.getLoc(), adaptor.getName());

    rewriter.replaceOp(op, adaptor.getMemory());
    return mlir::success();
  }
};

struct FloatToF64OpConversion
    : public ConvertPylirOpToLLVMPattern<Py::FloatToF64> {
  using ConvertPylirOpToLLVMPattern<
      Py::FloatToF64>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::FloatToF64 op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, pyFloatModel(rewriter, adaptor.getInput())
                               .doubleValue(op.getLoc())
                               .load(op.getLoc()));
    return mlir::success();
  }
};

struct ArithmeticSelectOpConversion
    : public ConvertPylirOpToLLVMPattern<mlir::arith::SelectOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    if (!isa<Py::DynamicType, Mem::MemoryType>(op.getType()))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

struct UnreachableOpConversion
    : public ConvertPylirOpToLLVMPattern<Py::UnreachableOp> {
  using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::UnreachableOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

struct MROLookupOpConversion : ConvertPylirOpToLLVMPattern<Py::MROLookupOp> {
  using ConvertPylirOpToLLVMPattern<
      Py::MROLookupOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::MROLookupOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto tuple = adaptor.getMroTuple();
    auto* block = op->getBlock();
    auto* endBlock = block->splitBlock(op);
    endBlock->addArguments(typeConverter.convertType(op.getType()), loc);

    rewriter.setInsertionPointToEnd(block);
    auto tupleSize = pyTupleModel(rewriter, tuple).size(loc).load(loc);
    auto startConstant =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, getIndexType(), 0);
    auto* conditionBlock = new mlir::Block;
    conditionBlock->addArgument(getIndexType(), loc);
    rewriter.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{startConstant},
                                      conditionBlock);

    conditionBlock->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(conditionBlock);
    auto isLess = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ult, conditionBlock->getArgument(0),
        tupleSize);
    auto* body = new mlir::Block;
    mlir::Value unbound = rewriter.create<mlir::LLVM::ZeroOp>(
        loc, endBlock->getArgument(0).getType());
    rewriter.create<mlir::LLVM::CondBrOp>(loc, isLess, body, endBlock, unbound);

    body->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(body);
    auto entry =
        unrealizedConversion(rewriter,
                             pyTupleModel(rewriter, tuple)
                                 .trailingArray(loc)
                                 .at(loc, conditionBlock->getArgument(0))
                                 .load(loc),
                             rewriter.getType<Py::DynamicType>());
    mlir::Value fetch = unrealizedConversion(
        rewriter, rewriter.create<Py::GetSlotOp>(loc, entry, adaptor.getSlot()),
        typeConverter);
    auto failure = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, fetch, unbound);
    auto* notFound = new mlir::Block;
    rewriter.create<mlir::LLVM::CondBrOp>(loc, failure, notFound, endBlock,
                                          fetch);

    notFound->insertBefore(endBlock);
    rewriter.setInsertionPointToStart(notFound);
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(loc, getIndexType(), 1);
    auto nextIter = rewriter.create<mlir::LLVM::AddOp>(
        loc, conditionBlock->getArgument(0), one);
    rewriter.create<mlir::LLVM::BrOp>(loc, mlir::ValueRange{nextIter},
                                      conditionBlock);

    rewriter.replaceOp(op, endBlock->getArguments());
    return mlir::success();
  }
};

struct FuncOpConversion : ConvertPylirOpToLLVMPattern<Py::FuncOp> {
  using ConvertPylirOpToLLVMPattern<Py::FuncOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::FuncOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::External;
    if (!op.isPublic())
      linkage = mlir::LLVM::Linkage::Internal;

    // TODO: It might be required to be doing function type/argument conversions
    // here in the future based on ABI.
    mlir::TypeConverter::SignatureConversion result(op.getNumArguments());
    mlir::Type llvmType = typeConverter.convertFunctionSignature(
        op.getFunctionType(), /*isVariadic=*/false,
        /*useBarePtrCallConv=*/false, result);

    auto newFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        op.getLoc(), op.getName(), llvmType, linkage,
        /*dsoLocal=*/true);
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    if (mlir::failed(rewriter.convertRegionTypes(&newFunc.getBody(),
                                                 typeConverter, &result)))
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReturnOpConversion : ConvertPylirOpToLLVMPattern<Py::ReturnOp> {
  using ConvertPylirOpToLLVMPattern<Py::ReturnOp>::ConvertPylirOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Py::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op,
                                                      adaptor.getArguments());
    return mlir::success();
  }
};

class ConvertPylirToLLVMPass
    : public pylir::impl::ConvertPylirToLLVMPassBase<ConvertPylirToLLVMPass> {
protected:
  void runOnOperation() override;

public:
  using Base::Base;
};
} // namespace

/// Explicitly checks the preconditions of this pass, returning failure and
/// emitting and error message otherwise.
[[maybe_unused]] static LogicalResult
checkPassPreconditions(ModuleOp moduleOp) {
  AttrTypeWalker walker;

  Operation* currentOperation;

  walker.addWalk([&](ObjectAttrInterface objectAttrInterface) {
    auto typeAttr =
        dyn_cast<TypeAttrInterface>(objectAttrInterface.getTypeObject());
    if (typeAttr)
      return WalkResult::advance();

    currentOperation->emitError("type-object of '")
        << objectAttrInterface << "' is not an instance of '"
        << Py::TypeAttr::getMnemonic() << "'";
    return WalkResult::interrupt();
  });

  WalkResult result = moduleOp.walk([&](Operation* op) {
    currentOperation = op;
    return walker.walk(op->getAttrDictionary());
  });

  return failure(/*isFailure=*/result.wasInterrupted());
}

namespace {

void ConvertPylirToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();

#ifndef NDEBUG
  if (failed(checkPassPreconditions(module)))
    return signalPassFailure();
#endif

  PylirTypeConverter converter(&getContext(), llvm::Triple(m_targetTripleCLI),
                               llvm::DataLayout(m_dataLayoutCLI),
                               mlir::DataLayout(module));
  CodeGenState codeGenState(converter, module);

  mlir::LLVMConversionTarget conversionTarget(getContext());
  conversionTarget
      .addIllegalDialect<Py::PylirPyDialect, Mem::PylirMemDialect>();
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  mlir::RewritePatternSet patternSet(&getContext());
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patternSet);
  patternSet.insert<
      ConstantOpConversion, ExternalOpConversion, GlobalOpConversion,
      StoreOpConversion, LoadOpConversion, IsOpConversion,
      IsUnboundValueOpConversion, TypeOfOpConversion, TupleGetItemOpConversion,
      TupleLenOpConversion, GetSlotOpConversion, SetSlotOpConversion,
      StrEqualOpConversion, StackAllocObjectOpConversion,
      GCAllocObjectOpConversion, InitObjectOpConversion, InitListOpConversion,
      InitTupleOpConversion, InitTupleFromListOpConversion, ListLenOpConversion,
      ListGetItemOpConversion, ListSetItemOpConversion, ListResizeOpConversion,
      RaiseOpConversion, InitIntUnsignedOpConversion, InitIntSignedOpConversion,
      ObjectHashOpConversion, ObjectIdOpConversion, StrHashOpConversion,
      InitFuncOpConversion, InitDictOpConversion, DictTryGetItemOpConversion,
      DictSetItemOpConversion, DictDelItemOpConversion, DictLenOpConversion,
      InitStrOpConversion, PrintOpConversion, InitStrFromIntOpConversion,
      InvokeOpsConversion<Py::InvokeOp>,
      InvokeOpsConversion<Py::FunctionInvokeOp>, CallOpConversion,
      FunctionCallOpConversion, BoolToI1OpConversion,
      InitTuplePrependOpConversion, InitTupleDropFrontOpConversion,
      IntToIndexOpConversion, IntCmpOpConversion, InitIntAddOpConversion,
      UnreachableOpConversion, TypeMROOpConversion,
      ArithmeticSelectOpConversion, TupleContainsOpConversion,
      InitTupleCopyOpConversion, MROLookupOpConversion, TypeSlotsOpConversion,
      InitFloatOpConversion, FloatToF64OpConversion, FuncOpConversion,
      ReturnOpConversion, RaiseExOpConversion, GCAllocFunctionOpConversion,
      FunctionGetClosureArgOpConversion, InitTypeOpConversion>(converter,
                                                               codeGenState);
  patternSet.insert<GCAllocObjectConstTypeConversion>(converter, codeGenState,
                                                      2);
  if (mlir::failed(mlir::applyFullConversion(module, conversionTarget,
                                             std::move(patternSet)))) {
    signalPassFailure();
    return;
  }
  auto builder = mlir::OpBuilder::atBlockEnd(module.getBody());
  builder.create<mlir::LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "pylir_personality_function",
      mlir::LLVM::LLVMFunctionType::get(
          builder.getI32Type(),
          {builder.getI32Type(), builder.getI64Type(),
           builder.getType<mlir::LLVM::LLVMPointerType>(),
           builder.getType<mlir::LLVM::LLVMPointerType>()}));
  for (auto iter : module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    iter.setGarbageCollectorAttr(
        mlir::StringAttr::get(&getContext(), "pylir-gc"));
    iter.setPersonalityAttr(mlir::FlatSymbolRefAttr::get(
        &getContext(), "pylir_personality_function"));
  }
  module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                  mlir::StringAttr::get(&getContext(), m_dataLayoutCLI));
  module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                  mlir::StringAttr::get(&getContext(), m_targetTripleCLI));
  if (auto globalInit = codeGenState.getGlobalInit()) {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToEnd(&globalInit.back());
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(),
                                         mlir::ValueRange{});

    builder.setInsertionPointToEnd(module.getBody());
    builder.create<mlir::LLVM::GlobalCtorsOp>(
        builder.getUnknownLoc(),
        builder.getArrayAttr({mlir::FlatSymbolRefAttr::get(globalInit)}),
        builder.getI32ArrayAttr({65535}));
  }
}
} // namespace
