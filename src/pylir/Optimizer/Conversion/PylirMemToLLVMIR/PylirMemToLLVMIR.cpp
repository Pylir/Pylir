#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Conversion/PassDetail.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "WinX64.hpp"
#include "X86_64.hpp"

namespace
{

class PylirTypeConverter : public mlir::LLVMTypeConverter
{
    llvm::DenseMap<pylir::Py::ObjectAttr, mlir::LLVM::GlobalOp> m_globalConstants;
    llvm::DenseMap<mlir::ArrayAttr, mlir::LLVM::GlobalOp> m_globalBuffers;
    mlir::SymbolTable m_symbolTable;

public:
    PylirTypeConverter(mlir::MLIRContext* context, const mlir::LowerToLLVMOptions& options, mlir::SymbolTable table)
        : mlir::LLVMTypeConverter(context, options), m_symbolTable(std::move(table))
    {
    }

    mlir::LLVM::LLVMStructType getPyObjectType()
    {
        auto pyObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyObject");
        if (!pyObject.isInitialized())
        {
            [[maybe_unused]] auto result = pyObject.setBody({mlir::LLVM::LLVMPointerType::get(pyObject)}, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyObject;
    }

    mlir::LLVM::LLVMStructType getBufferComponent()
    {
        auto component = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "BufferComponent");
        if (!component.isInitialized())
        {
            [[maybe_unused]] auto result = component.setBody(
                {getIndexType(), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType()))},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return component;
    }

    mlir::LLVM::LLVMStructType getPyTupleType()
    {
        auto pyTuple = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTuple");
        if (!pyTuple.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyTuple.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getBufferComponent()}, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyTuple;
    }

    mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttr objectAttr)
    {
        return llvm::TypeSwitch<pylir::Py::ObjectAttr, mlir::LLVM::LLVMStructType>(objectAttr)
            .Case([&](pylir::Py::TupleAttr) { return getPyTupleType(); })
            .Default([&](auto) { return getPyObjectType(); });
    }

    void initializeGlobal(mlir::LLVM::GlobalOp global, pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
    {
        builder.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), global.getType());
        auto typeObj = builder.create<mlir::LLVM::AddressOfOp>(
            global.getLoc(),
            mlir::LLVM::LLVMPointerType::get(
                typeOf(m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getValue()).initializer())),
            objectAttr.getType());
        auto bitcast = builder.create<mlir::LLVM::BitcastOp>(
            global.getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObjectType()), typeObj);
        undef =
            builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, bitcast, builder.getI32ArrayAttr({0}));
        llvm::TypeSwitch<pylir::Py::ObjectAttr>(objectAttr)
            .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr, pylir::Py::SetAttr>(
                [&](auto attr)
                {
                    auto values = attr.getValueAttr();
                    auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI64IntegerAttr(values.size()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 0}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 1}));
                    auto bufferObject = m_globalBuffers.lookup(values);
                    if (!bufferObject)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        bufferObject = builder.create<mlir::LLVM::GlobalOp>(
                            global.getLoc(),
                            mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                                           values.size()),
                            true, mlir::LLVM::Linkage::Private, "buffer$", mlir::Attribute{}, 0, 0, true);
                        bufferObject.setUnnamedAddrAttr(
                            mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                        m_symbolTable.insert(bufferObject);
                        m_globalBuffers.insert({values, bufferObject});
                        builder.setInsertionPointToStart(&bufferObject.getInitializerRegion().emplaceBlock());
                        mlir::Value arrayUndef =
                            builder.create<mlir::LLVM::UndefOp>(global.getLoc(), bufferObject.getType());
                        for (auto element : llvm::enumerate(values))
                        {
                            mlir::LLVM::AddressOfOp address;
                            if (auto ref = element.value().template dyn_cast<mlir::FlatSymbolRefAttr>())
                            {
                                address = builder.create<mlir::LLVM::AddressOfOp>(
                                    global.getLoc(),
                                    mlir::LLVM::LLVMPointerType::get(typeOf(
                                        m_symbolTable.lookup<pylir::Py::GlobalValueOp>(ref.getValue()).initializer())),
                                    ref);
                            }
                            else
                            {
                                address = builder.create<mlir::LLVM::AddressOfOp>(
                                    global.getLoc(),
                                    getConstant(element.value().template cast<pylir::Py::ObjectAttr>(), builder));
                            }
                            auto bitCast = builder.create<mlir::LLVM::BitcastOp>(
                                global.getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObjectType()), address);
                            arrayUndef = builder.create<mlir::LLVM::InsertValueOp>(
                                global.getLoc(), arrayUndef, bitCast,
                                builder.getI32ArrayAttr({static_cast<std::int32_t>(element.index())}));
                        }
                        builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), arrayUndef);
                    }
                    auto bufferAddress = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), bufferObject);
                    auto zero = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), builder.getI32Type(),
                                                                       builder.getI32IntegerAttr(0));
                    auto gep = builder.create<mlir::LLVM::GEPOp>(
                        global.getLoc(),
                        mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType())),
                        bufferAddress, mlir::ValueRange{zero, zero});
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, gep,
                                                                      builder.getI32ArrayAttr({1, 2}));
                })
            .Case(
                [&](pylir::Py::FloatAttr floatAttr)
                {
                    auto constant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), floatAttr.getValueAttr().getType(), floatAttr.getValueAttr());
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, constant,
                                                                      builder.getI32ArrayAttr({1}));
                })
            .Default(
                [](auto)
                {
                    // TODO: Not implemented yet
                    PYLIR_UNREACHABLE;
                });

        builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), undef);
    }

    mlir::LLVM::GlobalOp getConstant(pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
    {
        if (auto globalOp = m_globalConstants.lookup(objectAttr))
        {
            return globalOp;
        }
        mlir::OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
        auto type = typeOf(objectAttr);
        auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
            builder.getUnknownLoc(), type, true, mlir::LLVM::Linkage::Private, "const$", mlir::Attribute{}, 0, 0, true);
        globalOp.setUnnamedAddrAttr(mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
        m_symbolTable.insert(globalOp);
        m_globalConstants.insert({objectAttr, globalOp});
        initializeGlobal(globalOp, objectAttr, builder);
        return globalOp;
    }
};

template <class T>
struct ConvertPylirOpToLLVMPattern : public mlir::ConvertOpToLLVMPattern<T>
{
    explicit ConvertPylirOpToLLVMPattern(PylirTypeConverter& typeConverter, mlir::PatternBenefit benefit = 1)
        : mlir::ConvertOpToLLVMPattern<T>(typeConverter, benefit)
    {
    }

protected:
    PylirTypeConverter* getTypeConverter() const
    {
        return static_cast<PylirTypeConverter*>(this->typeConverter);
    }
};

struct ConstantOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ConstantOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ConstantOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::ConstantOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::ConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (auto unbound = adaptor.constant().dyn_cast<pylir::Py::UnboundAttr>())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
            return;
        }
        if (auto ref = adaptor.constant().dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, typeConverter->convertType(op.getType()), ref);
            return;
        }
        mlir::OpBuilder::InsertionGuard guard{rewriter};
        mlir::LLVM::GlobalOp globalOp =
            getTypeConverter()->getConstant(adaptor.constant().cast<pylir::Py::ObjectAttr>(), rewriter);
        auto addressOf = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), globalOp);
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), addressOf);
    }
};

struct GlobalValueOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GlobalValueOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GlobalValueOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::GlobalValueOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::GlobalValueOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto type = getTypeConverter()->typeOf(op.initializer());
        mlir::LLVM::Linkage linkage; // TODO: externally_available
        switch (op.getVisibility())
        {
            case mlir::SymbolTable::Visibility::Public: linkage = mlir::LLVM::linkage::Linkage::External; break;
            case mlir::SymbolTable::Visibility::Private: linkage = mlir::LLVM::linkage::Linkage::Private; break;
            case mlir::SymbolTable::Visibility::Nested: PYLIR_UNREACHABLE;
        }
        auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(op, type, op.constant(), linkage, op.getName(),
                                                                        mlir::Attribute{});
        getTypeConverter()->initializeGlobal(global, op.initializer(), rewriter);
    }
};

struct GlobalHandleOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GlobalHandleOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GlobalHandleOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::GlobalHandleOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::GlobalHandleOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::LLVM::Linkage linkage;
        switch (op.getVisibility())
        {
            case mlir::SymbolTable::Visibility::Public: linkage = mlir::LLVM::linkage::Linkage::External; break;
            case mlir::SymbolTable::Visibility::Private: linkage = mlir::LLVM::linkage::Linkage::Private; break;
            case mlir::SymbolTable::Visibility::Nested: PYLIR_UNREACHABLE;
        }
        auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
            op, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()), false, linkage, op.getName(),
            mlir::Attribute{});
        rewriter.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), global.getType());
        rewriter.create<mlir::LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange{null});
    }
};

struct LoadOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::LoadOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::LoadOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::LoadOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::LoadOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            op.handleAttr());
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, address);
    }
};

struct StoreOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StoreOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StoreOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::StoreOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            adaptor.handle());
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), address);
    }
};

struct IsOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IsOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IsOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::IsOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::IsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::eq, adaptor.lhs(),
                                                        adaptor.rhs());
    }
};

struct IsUnboundValueOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IsUnboundValueOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IsUnboundValueOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::IsUnboundValueOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::IsUnboundValueOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), adaptor.value().getType());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::eq, adaptor.value(), null);
    }
};

struct TypeOfOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TypeOfOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TypeOfOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::TypeOfOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::TypeOfOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            adaptor.object(), mlir::ValueRange{zero, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
    }
};

struct TupleIntegerGetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleIntegerGetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleIntegerGetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::TupleIntegerGetItemOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::TupleIntegerGetItemOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTupleType()), adaptor.tuple());
        auto bufferStartPtr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(
                mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()))),
            tuple, mlir::ValueRange{zero, one, two});
        auto bufferStart = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), bufferStartPtr);
        auto offset =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), bufferStart.getType(), bufferStart, adaptor.index());
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, offset);
    }
};

struct TupleIntegerLenOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleIntegerLenOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleIntegerLenOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::TupleIntegerLenOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::TupleIntegerLenOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTupleType()), adaptor.tuple());
        auto sizePtr = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                          tuple, mlir::ValueRange{zero, one, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, sizePtr);
    }
};

struct ConvertPylirToLLVMPass : public pylir::ConvertPylirToLLVMBase<ConvertPylirToLLVMPass>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirToLLVMPass::runOnOperation()
{
    auto module = getOperation();

    mlir::LowerToLLVMOptions options(&getContext());
    options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::None;
    options.dataLayout = llvm::DataLayout(dataLayout.getValue());
    PylirTypeConverter converter(&getContext(), options, mlir::SymbolTable(module));
    converter.addConversion([&](pylir::Py::DynamicType)
                            { return mlir::LLVM::LLVMPointerType::get(converter.getPyObjectType()); });
    converter.addConversion([&](pylir::Mem::MemoryType)
                            { return mlir::LLVM::LLVMPointerType::get(converter.getPyObjectType()); });

    mlir::LLVMConversionTarget conversionTarget(getContext());
    conversionTarget.addIllegalDialect<pylir::Py::PylirPyDialect, pylir::Mem::PylirMemDialect>();
    conversionTarget.addLegalOp<mlir::ModuleOp>();

    mlir::RewritePatternSet patternSet(&getContext());
    mlir::populateStdToLLVMConversionPatterns(converter, patternSet);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(converter, patternSet);
    mlir::populateReconcileUnrealizedCastsPatterns(patternSet);
    patternSet.insert<ConstantOpConversion>(converter);
    patternSet.insert<GlobalValueOpConversion>(converter);
    patternSet.insert<GlobalHandleOpConversion>(converter);
    patternSet.insert<StoreOpConversion>(converter);
    patternSet.insert<LoadOpConversion>(converter);
    patternSet.insert<IsOpConversion>(converter);
    patternSet.insert<IsUnboundValueOpConversion>(converter);
    patternSet.insert<TypeOfOpConversion>(converter);
    patternSet.insert<TupleIntegerGetItemOpConversion>(converter);
    patternSet.insert<TupleIntegerLenOpConversion>(converter);
    if (mlir::failed(mlir::applyFullConversion(module, conversionTarget, std::move(patternSet))))
    {
        signalPassFailure();
        return;
    }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Mem::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
