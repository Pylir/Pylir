
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

mlir::Value PyLong_FromSize_t(mlir::OpBuilder& rewriter, mlir::Location location, mlir::Value constant,
                              mlir::ModuleOp module)
{
    auto func = declareFunc(
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMPointerType::get(getPyLongObject(rewriter.getContext())),
                                          {rewriter.getI64Type()}),
        "PyLong_FromSize_t", module);
    return rewriter.create<mlir::LLVM::CallOp>(location, func, mlir::ValueRange{constant}).getResult(0);
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
                    // TODO larger than index using a pylir private function
                    auto constant = rewriter.create<mlir::LLVM::ConstantOp>(
                        op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(value.getValue().getSExtValue()));

                    rewriter.replaceOp(op, {PyLong_FromSize_t(rewriter, op->getLoc(), constant,
                                                              op->getParentOfType<mlir::ModuleOp>())});
                })
            .Case<mlir::FlatSymbolRefAttr>(
                [](auto)
                {
                    // TODO
                    PYLIR_UNREACHABLE;
                })
            .Case<pylir::Dialect::DictAttr>(
                [&](auto)
                {
                    // TODO, null for now
                    rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(
                        op, mlir::LLVM::LLVMPointerType::get(getPyDictObject(rewriter.getContext())));
                })
            .Default([](auto) { PYLIR_UNREACHABLE; });
    }
};

struct ConstantGlobalOpConversion : SingleOpMatcher<ConstantGlobalOpConversion, pylir::Dialect::ConstantGlobalOp>
{
    std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors;

    ConstantGlobalOpConversion(std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors,
                               llvm::StringRef rootOpName, mlir::MLIRContext* context,
                               mlir::LLVMTypeConverter& typeConverter, mlir::PatternBenefit benefit = 1)
        : SingleOpMatcher(rootOpName, context, typeConverter, benefit), ctors(ctors)
    {
    }

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
                mlir::LLVM::LLVMFunctionType::get(rewriter.getType<mlir::LLVM::LLVMVoidType>(), {}), newOp.linkage());
            rewriter.setInsertionPointToStart(ctor.addEntryBlock());
            auto address = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), newOp);
            auto zero =
                rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(0));
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
            auto one =
                rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
            auto dict = rewriter.create<mlir::LLVM::GEPOp>(
                op->getLoc(),
                mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8))), address,
                mlir::ValueRange{zero, one});
            // set to nullptr for now
            auto null = rewriter.create<mlir::LLVM::NullOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), null, dict);
        }

        ctors.emplace_back(ctor, newOp);
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
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
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
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(0));
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
        auto input = adaptor.input();
        if (!mlir::cast<pylir::Dialect::GetAttrOp>(op).input().getType().isa<pylir::Dialect::UnknownType>())
        {
            input = rewriter.create<mlir::LLVM::BitcastOp>(
                op->getLoc(), typeConverter->convertType(rewriter.getType<pylir::Dialect::UnknownType>()), input);
        }
        mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
        mlir::LLVM::GlobalOp stringOp;
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        {
            auto str = adaptor.member().getValue();
            rewriter.setInsertionPoint(moduleOp.getBody()->getTerminator());
            std::string name = ".str";
            if (moduleOp.lookupSymbol(name))
            {
                auto copy = name;
                std::size_t counter = 0;
                do
                {
                    name = copy + "." + std::to_string(counter);
                    counter++;
                } while (moduleOp.lookupSymbol(name));
            }
            stringOp = rewriter.create<mlir::LLVM::GlobalOp>(
                rewriter.getUnknownLoc(), mlir::LLVM::LLVMArrayType::get(rewriter.getIntegerType(8), str.size()), true,
                mlir::LLVM::Linkage::Internal, name, adaptor.member());
            // stringOp->setAttr("unnamed_addr",mlir::LLVM::UnnamedAddrAttr::get(rewriter.getContext()));
        }
        auto ptr = mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext()));
        auto i8Star = mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));
        auto func = declareFunc(mlir::LLVM::LLVMFunctionType::get(ptr, {ptr, i8Star, rewriter.getI64Type()}),
                                "pylir_get_attr", moduleOp);
        auto size = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                            rewriter.getIndexAttr(adaptor.member().getValue().size()));
        auto str = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), stringOp);
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(0));
        auto call = rewriter.create<mlir::LLVM::CallOp>(
            op->getLoc(), func,
            mlir::ValueRange{
                input, rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), i8Star, str, mlir::ValueRange{zero, zero}),
                size});
        auto null = rewriter.create<mlir::LLVM::NullOp>(op->getLoc(), call.getType(0));
        auto found =
            rewriter.create<mlir::LLVM::ICmpOp>(op->getLoc(), mlir::LLVM::ICmpPredicate::ne, call.getResult(0), null);
        rewriter.replaceOp(op, {call.getResult(0), found});
    }
};

struct AllocaConversion : SingleOpMatcher<AllocaConversion, pylir::Dialect::AllocaOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::AllocaOp::Adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto size =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
            op, typeConverter->convertType(rewriter.getType<pylir::Dialect::HandleType>()), size, 8);
    }
};

struct IdConversion : SingleOpMatcher<IdConversion, pylir::Dialect::IdOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::IdOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto id = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), rewriter.getI64Type(), adaptor.input());
        rewriter.replaceOp(op, {PyLong_FromSize_t(rewriter, op->getLoc(), id, op->getParentOfType<mlir::ModuleOp>())});
    }
};

struct ICmpConversion : SingleOpMatcher<ICmpConversion, pylir::Dialect::ICmpOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ICmpOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        std::string_view funcName;
        switch (mlir::cast<pylir::Dialect::ICmpOp>(op).predicate())
        {
            case pylir::Dialect::CmpPredicate::EQ: funcName = "pylir_icmp_eq"; break;
            case pylir::Dialect::CmpPredicate::NE: funcName = "pylir_icmp_ne"; break;
            case pylir::Dialect::CmpPredicate::LT: funcName = "pylir_icmp_lt"; break;
            case pylir::Dialect::CmpPredicate::LE: funcName = "pylir_icmp_le"; break;
            case pylir::Dialect::CmpPredicate::GT: funcName = "pylir_icmp_gt"; break;
            case pylir::Dialect::CmpPredicate::GE: funcName = "pylir_icmp_ge"; break;
            default: PYLIR_UNREACHABLE;
        }
        auto iPtr = mlir::LLVM::LLVMPointerType::get(getPyLongObject(rewriter.getContext()));
        auto func = declareFunc(
            mlir::LLVM::LLVMFunctionType::get(typeConverter->convertType(op->getResultTypes()[0]), {iPtr, iPtr}),
            funcName, op->getParentOfType<mlir::ModuleOp>());
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, func, mlir::ValueRange{adaptor.lhs(), adaptor.rhs()});
    }
};

struct Bto1Conversion : SingleOpMatcher<Bto1Conversion, pylir::Dialect::BtoI1Op>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::BtoI1Op::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto ptr = typeConverter->convertType(rewriter.getType<pylir::Dialect::BoolType>());
        auto func = declareFunc(mlir::LLVM::LLVMFunctionType::get(rewriter.getI1Type(), {ptr}), "pylir_bTo1",
                                op->getParentOfType<mlir::ModuleOp>());
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, func, mlir::ValueRange{adaptor.input()});
    }
};

struct MakeTupleConversion : SingleOpMatcher<MakeTupleConversion, pylir::Dialect::MakeTupleOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::MakeTupleOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto pyObjectStar = mlir::LLVM::LLVMPointerType::get(getPyObject(rewriter.getContext()));
        auto arrayType = mlir::LLVM::LLVMArrayType::get(pyObjectStar, adaptor.input().size());
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(1));
        auto alloca =
            rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(arrayType), one, 8);
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getIndexAttr(0));
        auto pyObjectStarStar = mlir::LLVM::LLVMPointerType::get(pyObjectStar);
        for (auto iter = adaptor.input().begin(); iter != adaptor.input().end(); iter++)
        {
            auto constant = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                                    rewriter.getIndexAttr(iter.getIndex()));
            auto gep = rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), pyObjectStarStar, alloca,
                                                          mlir::ValueRange{zero, constant});
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), *iter, gep);
        }
        auto func = declareFunc(
            mlir::LLVM::LLVMFunctionType::get(typeConverter->convertType(rewriter.getType<pylir::Dialect::TupleType>()),
                                              {pyObjectStarStar, rewriter.getI64Type()}),
            "pylir_make_tuple", op->getParentOfType<mlir::ModuleOp>());
        auto size = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                            rewriter.getIndexAttr(adaptor.input().size()));
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
            op, func,
            mlir::ValueRange{rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), pyObjectStarStar, alloca,
                                                                mlir::ValueRange{zero, zero}),
                             size});
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
        [](pylir::Dialect::KnownTypeObjectType type) -> llvm::Optional<mlir::Type>
        {
            auto value = type.getType().getValue();
            if (value == llvm::StringRef{pylir::Dialect::noneTypeObjectName}
                || value == llvm::StringRef{pylir::Dialect::notImplementedTypeObjectName})
            {
                return mlir::LLVM::LLVMPointerType::get(getPyObject(type.getContext()));
            }
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

    converter.addConversion(
        [](pylir::Dialect::BoolType type) -> llvm::Optional<mlir::Type>
        {
            // TODO: PyBoolObject
            return getPyLongObject(type.getContext());
        });

    populateSingleOpMatchers<GlobalOpConversion, ReinterpretOpConversion, ConstantOpConversion,
                             GetFunctionPointerConversion, GetItemConversion, IMulConversion, HandleOfConversion,
                             DataOfConversion, LoadConversion, StoreConversion, TypeOfConversion, GetAttrConversion,
                             AllocaConversion, IdConversion, ICmpConversion, Bto1Conversion, MakeTupleConversion>(
        converter, patterns);
    std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>> ctors;
    patterns.insert<ConstantGlobalOpConversion>(ctors, ConstantGlobalOpConversion::Operation::getOperationName(),
                                                &converter.getContext(), converter);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<pylir::Dialect::PylirDialect, mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
        return;
    }

    // Add global ctors to llvm.global_ctors
    mlir::OpBuilder builder{module.getBody()->getTerminator()};
    auto literal = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {builder.getI32Type(),
                               mlir::LLVM::LLVMPointerType::get(
                                   mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(), {})),
                               mlir::LLVM::LLVMPointerType::get(builder.getIntegerType(8))});
    auto append = builder.create<mlir::LLVM::GlobalOp>(
        builder.getUnknownLoc(), mlir::LLVM::LLVMArrayType::get(literal, ctors.size()), false,
        mlir::LLVM::Linkage::Appending, "llvm.global_ctors", mlir::Attribute{});
    append->setAttr("section", builder.getStringAttr("llvm.metadata"));

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto* block = new mlir::Block;
    append.getInitializerRegion().push_back(block);
    builder.setInsertionPointToStart(block);

    mlir::Value emptyArray = builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), append.getType());

    std::size_t count = 0;
    for (auto& [ctor, op] : ctors)
    {
        mlir::Value emptyStruct = builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), literal);
        auto priority = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(),
                                                               builder.getI32IntegerAttr(65535));
        emptyStruct = builder.create<mlir::LLVM::InsertValueOp>(builder.getUnknownLoc(), emptyStruct, priority,
                                                                builder.getI64ArrayAttr({0}));
        auto constructorAddress = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), ctor);
        emptyStruct = builder.create<mlir::LLVM::InsertValueOp>(builder.getUnknownLoc(), emptyStruct,
                                                                constructorAddress, builder.getI64ArrayAttr({1}));
        auto dataAddress = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), op);
        emptyStruct = builder.create<mlir::LLVM::InsertValueOp>(builder.getUnknownLoc(), emptyStruct, dataAddress,
                                                                builder.getI64ArrayAttr({2}));
        emptyArray =
            builder.create<mlir::LLVM::InsertValueOp>(builder.getUnknownLoc(), emptyArray, emptyStruct,
                                                      builder.getI64ArrayAttr({static_cast<std::int64_t>(count)}));
        count++;
    }
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{emptyArray});
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Dialect::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
