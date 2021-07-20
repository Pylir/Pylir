
#include "PylirToLLVM.hpp"

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Dialect/PylirDialect.hpp>
#include <pylir/Optimizer/Dialect/PylirOps.hpp>
#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>
#include <pylir/Support/Macros.hpp>

#include <optional>

#include "PassDetail.hpp"

namespace
{
mlir::Type getPyObject(mlir::MLIRContext* context)
{
    auto pyObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyObject");
    if (pyObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(mlir::LLVM::LLVMPointerType::get(pyObject)); // pointer to the type
        pyObject.setBody(types, false);
    }
    return pyObject;
}

mlir::Type getPyBaseObject(mlir::MLIRContext* context)
{
    auto pyBaseObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyBaseObject");
    if (pyBaseObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyObject(context)); // base
        types.emplace_back(
            mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8))); // opaque pointer to the dict for now
        pyBaseObject.setBody(types, false);
    }
    return pyBaseObject;
}

mlir::Type getPyTupleObject(mlir::MLIRContext* context)
{
    auto pyTupleObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyTupleObject");
    if (pyTupleObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyObject(context)); // base
        types.emplace_back(mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(context, 8))); // opaque pointer to the tuple for now
        pyTupleObject.setBody(types, false);
    }
    return pyTupleObject;
}

mlir::Type getPyDictObject(mlir::MLIRContext* context)
{
    auto pyDictObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyDictObject");
    if (pyDictObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyObject(context)); // base
        types.emplace_back(
            mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8))); // opaque pointer to the dict for now
        pyDictObject.setBody(types, false);
    }
    return pyDictObject;
}

mlir::Type getPyFunctionObject(mlir::MLIRContext* context)
{
    auto pyFunctionObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyFunctionObject");
    if (pyFunctionObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyBaseObject(context)); // base
        auto pyObject = getPyObject(context);
        types.emplace_back(mlir::LLVM::LLVMPointerType::get(
            mlir::LLVM::LLVMFunctionType::get(pyObject, {pyObject, pyObject, pyObject})));
        pyFunctionObject.setBody(types, false);
    }
    return pyFunctionObject;
}

mlir::Type getPyLongObject(mlir::MLIRContext* context)
{
    auto pyLongObject = mlir::LLVM::LLVMStructType::getIdentified(context, "PyLongObject");
    if (pyLongObject.getBody().empty())
    {
        std::vector<mlir::Type> types;
        types.emplace_back(getPyObject(context)); // base
        types.emplace_back(mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(context, 8))); // opaque pointer to the integer for now
        pyLongObject.setBody(types, false);
    }
    return pyLongObject;
}

mlir::LLVM::LLVMFuncOp declareFunc(mlir::LLVM::LLVMFunctionType functionType, std::string_view name,
                                   mlir::ModuleOp module, std::optional<mlir::ArrayAttr> passthrough = {})
{
    if (auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    {
        PYLIR_ASSERT(funcOp.getType() == functionType);
        return funcOp;
    }
    mlir::OpBuilder builder(module.getContext());
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, functionType);
    if (passthrough)
    {
        funcOp.passthroughAttr(*passthrough);
    }
    module.push_back(funcOp);
    return funcOp;
}

template <class Self, class Op>
struct SingleOpMatcher : public mlir::ConversionPattern
{
    using Operation = Op;

    SingleOpMatcher(llvm::StringRef rootOpName, mlir::MLIRContext* context, mlir::LLVMTypeConverter& typeConverter,
                    mlir::PatternBenefit benefit = 1)
        : mlir::ConversionPattern(rootOpName, benefit, typeConverter, context)
    {
    }

    mlir::LogicalResult match(mlir::Operation* op) const override
    {
        return mlir::success(mlir::isa<Op>(op));
    }

    void rewrite(mlir::Operation* op, llvm::ArrayRef<mlir::Value> operands,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        static_cast<const Self*>(this)->Self::rewrite(op, typename Op::Adaptor(operands, op->getAttrDictionary()),
                                                      rewriter);
    }
};

struct ConstantOpConversion : SingleOpMatcher<ConstantOpConversion, pylir::Dialect::ConstantOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ConstantOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        llvm::TypeSwitch<mlir::Attribute>(adaptor.value())
            .Case<pylir::Dialect::IntegerAttr>(
                [&](pylir::Dialect::IntegerAttr value)
                {
                    // TODO larger than long using a pylir private function
                    auto func =
                        declareFunc(mlir::LLVM::LLVMFunctionType::get(
                                        mlir::LLVM::LLVMPointerType::get(getPyLongObject(rewriter.getContext())),
                                        {/*TODO: should be C long*/ rewriter.getI64Type()}),
                                    "PyLong_FromLong", op->getParentOfType<mlir::ModuleOp>());
                    auto constant = rewriter.create<mlir::LLVM::ConstantOp>(
                        op->getLoc(), rewriter.getI64Type(),
                        rewriter.getI64IntegerAttr(value.getValue().getSExtValue()));
                    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, func, mlir::ValueRange{constant});
                })
            .Case<mlir::FlatSymbolRefAttr>(
                [](auto)
                {
                    // TODO
                    PYLIR_UNREACHABLE;
                })
            .Default([](auto) { PYLIR_UNREACHABLE; });
    }
};

struct ConstantGlobalOpConversion : SingleOpMatcher<ConstantGlobalOpConversion, pylir::Dialect::ConstantGlobalOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ConstantGlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto exit = llvm::make_scope_exit([&] { rewriter.eraseOp(op); });
        auto resultType = typeConverter->convertType(adaptor.type().getValue());
        PYLIR_ASSERT(resultType);
        auto newOp =
            rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType, false, mlir::LLVM::Linkage::LinkonceODR,
                                                  adaptor.sym_name().getValue(), mlir::Attribute{});
        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            auto* block = new mlir::Block;
            newOp.getInitializerRegion().push_back(block);
            rewriter.setInsertionPointToStart(block);
            auto undef = rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), newOp.getType());
            rewriter.create<mlir::LLVM::ReturnOp>(op->getLoc(), mlir::ValueRange{undef});
        }

        mlir::LLVM::LLVMFuncOp ctor;

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            ctor = rewriter.create<mlir::LLVM::LLVMFuncOp>(
                op->getLoc(), (adaptor.sym_name().getValue() + ".ctor").str(),
                mlir::LLVM::LLVMFunctionType::get(rewriter.getType<mlir::LLVM::LLVMVoidType>(), {}),
                op->hasAttr("linkonce") ? mlir::LLVM::Linkage::LinkonceODR : mlir::LLVM::Linkage::Internal);
            rewriter.setInsertionPointToStart(ctor.addEntryBlock());
            auto address = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), newOp);
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIndexType(),
                                                                rewriter.getIndexAttr(0));
            auto typePointer = rewriter.create<mlir::LLVM::GEPOp>(
                op->getLoc(),
                mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext()))),
                address, mlir::ValueRange{zero, zero, zero});
            mlir::Value typeObject = rewriter.create<mlir::LLVM::AddressOfOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getPyBaseObject(rewriter.getContext())),
                adaptor.type().getValue().cast<pylir::Dialect::KnownTypeObjectType>().getType());
            typeObject = rewriter.create<mlir::LLVM::GEPOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext())), typeObject,
                mlir::ValueRange{zero, zero});
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), typePointer, typeObject);
            // TODO init
            auto one = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIndexType(),
                                                               rewriter.getIndexAttr(1));
            auto dict = rewriter.create<mlir::LLVM::GEPOp>(
                op->getLoc(),
                mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8))), address,
                mlir::ValueRange{zero, one});
            // set to nullptr for now
            auto null = rewriter.create<mlir::LLVM::NullOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), null, dict);
        }
        auto literal = mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {rewriter.getI32Type(), mlir::LLVM::LLVMPointerType::get(ctor.getType()),
                                    mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8))});
        auto append = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), mlir::LLVM::LLVMArrayType::get(literal, 1),
                                                            false, mlir::LLVM::Linkage::Appending, "llvm.global_ctors",
                                                            mlir::Attribute{});
        append->setAttr("section", rewriter.getStringAttr("llvm.metadata"));

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto* block = new mlir::Block;
        append.getInitializerRegion().push_back(block);
        rewriter.setInsertionPointToStart(block);
        mlir::Value emptyStruct = rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), literal);
        auto priority = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                                                rewriter.getI32IntegerAttr(65535));
        emptyStruct = rewriter.create<mlir::LLVM::InsertValueOp>(op->getLoc(), emptyStruct, priority,
                                                                 rewriter.getIndexArrayAttr({0}));
        auto constructorAddress = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), ctor);
        emptyStruct = rewriter.create<mlir::LLVM::InsertValueOp>(op->getLoc(), emptyStruct, constructorAddress,
                                                                 rewriter.getIndexArrayAttr({1}));
        auto dataAddress = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), newOp);
        emptyStruct = rewriter.create<mlir::LLVM::InsertValueOp>(op->getLoc(), emptyStruct, dataAddress,
                                                                 rewriter.getIndexArrayAttr({2}));
        auto emptyArray = rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), append.getType());
        rewriter.create<mlir::LLVM::ReturnOp>(
            op->getLoc(), mlir::ValueRange{rewriter.create<mlir::LLVM::InsertValueOp>(
                              op->getLoc(), emptyArray, emptyStruct, rewriter.getIndexArrayAttr({0}))});
    }
};

struct GlobalOpConversion : SingleOpMatcher<GlobalOpConversion, pylir::Dialect::GlobalOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto exit = llvm::make_scope_exit([&] { rewriter.eraseOp(op); });
        mlir::Type resultType = mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext()));
        auto newOp =
            rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType, false, mlir::LLVM::Linkage::External,
                                                  adaptor.sym_name().getValue(), mlir::Attribute{});
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto* block = new mlir::Block;
        newOp.getInitializerRegion().push_back(block);
        rewriter.setInsertionPointToStart(block);
        auto undef = rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), newOp.getType());
        rewriter.create<mlir::LLVM::ReturnOp>(op->getLoc(), mlir::ValueRange{undef});
    }
};

struct ReinterpretOpConversion : SingleOpMatcher<ReinterpretOpConversion, pylir::Dialect::ReinterpretOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ReinterpretOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op->getResultTypes().front()),
                                                           adaptor.input());
    }
};

struct GetFunctionPointerConversion
    : SingleOpMatcher<GetFunctionPointerConversion, pylir::Dialect::GetFunctionPointerOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetFunctionPointerOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(1));
        auto fpP = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op->getResultTypes()[0])),
            adaptor.input(), mlir::ValueRange{zero, one});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, fpP);
    }
};

struct GetItemConversion : SingleOpMatcher<GetItemConversion, pylir::Dialect::GetItemOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetItemOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto sequenceType = mlir::cast<pylir::Dialect::GetItemOp>(op).sequence().getType();
        llvm::TypeSwitch<mlir::Type>(sequenceType)
            .Case<pylir::Dialect::TupleType>(
                [&](auto)
                {
                    auto func = declareFunc(mlir::LLVM::LLVMFunctionType::get(
                                                mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext())),
                                                {/*TODO should be PyObject**/ mlir::LLVM::LLVMPointerType::get(
                                                     getPyTupleObject(rewriter.getContext())),
                                                 /*TODO should be size_t aka index*/ mlir::LLVM::LLVMPointerType::get(
                                                     getPyLongObject(rewriter.getContext()))}),
                                            "PyTuple_GetItem", op->getParentOfType<mlir::ModuleOp>());
                    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                        op, func, mlir::ValueRange{adaptor.sequence(), adaptor.index()});
                })
            .Default(
                [](auto)
                {
                    // TODO
                    PYLIR_UNREACHABLE;
                });
    }
};

struct IMulConversion : SingleOpMatcher<IMulConversion, pylir::Dialect::IMulOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::IMulOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto longPtr = mlir::LLVM::LLVMPointerType::get(getPyLongObject(rewriter.getContext()));
        auto func = declareFunc(mlir::LLVM::LLVMFunctionType::get(longPtr, {longPtr, longPtr}), "pylir_imul",
                                op->getParentOfType<mlir::ModuleOp>());
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, func, mlir::ValueRange{adaptor.lhs(), adaptor.rhs()});
    }
};

struct HandleOfConversion : SingleOpMatcher<HandleOfConversion, pylir::Dialect::HandleOfOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::HandleOfOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                             adaptor.globalName());
    }
};

struct DataOfConversion : SingleOpMatcher<DataOfConversion, pylir::Dialect::DataOfOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::DataOfOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                             adaptor.globalName());
    }
};

struct LoadConversion : SingleOpMatcher<LoadConversion, pylir::Dialect::LoadOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::LoadOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto load = rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), adaptor.handle());
        auto loadedType = op->getResultTypes()[0];
        if (loadedType.isa<pylir::Dialect::UnknownType>())
        {
            rewriter.replaceOp(op, {load});
            return;
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(loadedType), load);
    }
};

struct StoreConversion : SingleOpMatcher<StoreConversion, pylir::Dialect::StoreOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::StoreOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto toStoreType = mlir::cast<pylir::Dialect::StoreOp>(op).value().getType();
        auto toStore = adaptor.value();
        if (!toStoreType.isa<pylir::Dialect::UnknownType>())
        {
            toStore = rewriter.create<mlir::LLVM::BitcastOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext())), toStore);
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, toStore, adaptor.handle());
    }
};

struct TypeOfConversion : SingleOpMatcher<TypeOfConversion, pylir::Dialect::TypeOfOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::TypeOfOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto inputType = mlir::cast<pylir::Dialect::TypeOfOp>(op).input().getType();
        auto input = adaptor.input();
        if (!inputType.isa<pylir::Dialect::UnknownType>())
        {
            input = rewriter.create<mlir::LLVM::BitcastOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext())), input);
        }
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op->getResultTypes()[0])), input,
            mlir::ValueRange{zero, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
    }
};

struct GetAttrConversion : SingleOpMatcher<GetAttrConversion, pylir::Dialect::GetAttrOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetAttrOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
        mlir::LLVM::GlobalOp stringOp;
        {
            auto str = adaptor.member().getValue();
            auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
            rewriter.setInsertionPointToEnd(moduleOp.getBody());
            std::string name = ".str";
            bool first = true;
            while (moduleOp.lookupSymbol(name))
            {
                if (first)
                {
                    name += ".0";
                    first = false;
                    continue;
                }
            }
            stringOp = rewriter.create<mlir::LLVM::GlobalOp>(
                rewriter.getUnknownLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getIntegerType(8), str.size() + 1),
                true, mlir::LLVM::Linkage::Internal, , rewriter.getStringAttr((str + "\0").str()));
            // stringOp->setAttr("unnamed_addr",mlir::LLVM::UnnamedAddrAttr::get(rewriter.getContext()));
        }
    }
};

template <class... Args>
void populateSingleOpMatchers(mlir::LLVMTypeConverter& converter, mlir::OwningRewritePatternList& patterns)
{
    (patterns.insert<Args>(Args::Operation::getOperationName(), &converter.getContext(), converter), ...);
}

} // namespace

namespace
{
struct ConvertPylirToLLVMPass : public pylir::Dialect::ConvertPylirToLLVMBase<ConvertPylirToLLVMPass>
{
protected:
    void runOnOperation() override;
};

void ConvertPylirToLLVMPass::runOnOperation()
{
    auto module = getOperation();

    mlir::OwningRewritePatternList patterns;
    mlir::LLVMTypeConverter converter(&getContext());

    converter.addConversion(
        [](pylir::Dialect::UnknownType type) -> llvm::Optional<mlir::Type> {
            return mlir::LLVM::LLVMPointerType::get(getPyObject(type.getContext()));
        });
    converter.addConversion(
        [](pylir::Dialect::KnownTypeObjectType type) -> llvm::Optional<mlir::Type> {
            return mlir::LLVM::LLVMPointerType::get(getPyBaseObject(type.getContext()));
        });
    converter.addConversion(
        [](pylir::Dialect::FunctionType type) -> llvm::Optional<mlir::Type> {
            return getPyFunctionObject(type.getContext());
        });
    converter.addConversion(
        [](pylir::Dialect::TupleType type) -> llvm::Optional<mlir::Type> {
            return getPyTupleObject(type.getContext());
        });
    converter.addConversion(
        [](pylir::Dialect::HandleType type) -> llvm::Optional<mlir::Type> {
            return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getPyObject(type.getContext())));
        });

    converter.addConversion(
        [](pylir::Dialect::DictType type) -> llvm::Optional<mlir::Type> { return getPyDictObject(type.getContext()); });

    converter.addConversion(
        [](pylir::Dialect::IntegerType type) -> llvm::Optional<mlir::Type> {
            return getPyLongObject(type.getContext());
        });

    populateSingleOpMatchers<GlobalOpConversion, ConstantGlobalOpConversion, ReinterpretOpConversion,
                             ConstantOpConversion, GetFunctionPointerConversion, GetItemConversion, IMulConversion,
                             HandleOfConversion, DataOfConversion, LoadConversion, StoreConversion, TypeOfConversion,
                             GetAttrConversion>(converter, patterns);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<pylir::Dialect::PylirDialect, mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Dialect::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
