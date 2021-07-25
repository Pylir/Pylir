
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
class PylirTypeConverter : public mlir::LLVMTypeConverter
{
    llvm::DenseMap<llvm::StringRef, mlir::Type> m_knownObjectTypes;
    mlir::LLVM::LLVMStructType m_pyObject;
    mlir::LLVM::LLVMStructType m_pyTupleObject;
    mlir::LLVM::LLVMStructType m_pyDictObject;
    mlir::LLVM::LLVMStructType m_pyFunctionObject;
    mlir::LLVM::LLVMStructType m_pyLongObject;
    mlir::LLVM::LLVMStructType m_pyTypeObject;

    mlir::Type getTypeFromTypeObject(pylir::Dialect::ConstantGlobalOp constant)
    {
        // builtin cases first
        auto value = constant.type().cast<pylir::Dialect::KnownTypeObjectType>().getType().getValue();
        if (value == llvm::StringRef{pylir::Dialect::typeTypeObjectName})
        {
            return getPyTypeObject();
        }
        if (value == llvm::StringRef{pylir::Dialect::longTypeObjectName})
        {
            return getPyLongObject();
        }
        if (value == llvm::StringRef{pylir::Dialect::functionTypeObjectName})
        {
            return getPyFunctionObject();
        }
        // All others are just a PyObject with a dict
        return getPyObject();
    }

public:
    explicit PylirTypeConverter(mlir::ModuleOp module) : mlir::LLVMTypeConverter(module.getContext())
    {
        addConversion(
            [&](pylir::Dialect::UnknownType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(getPyObject());
            });
        addConversion(
            [&](pylir::Dialect::FunctionType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(getPyFunctionObject());
            });
        addConversion(
            [&](pylir::Dialect::TupleType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(getPyTupleObject());
            });
        addConversion(
            [&](pylir::Dialect::HandleType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getPyObject()));
            });

        addConversion(
            [&](pylir::Dialect::DictType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(getPyDictObject());
            });

        addConversion(
            [&](pylir::Dialect::IntegerType) -> llvm::Optional<mlir::Type> {
                return mlir::LLVM::LLVMPointerType::get(getPyLongObject());
            });

        addConversion(
            [&](pylir::Dialect::BoolType) -> llvm::Optional<mlir::Type>
            {
                // TODO: PyBoolObject
                return mlir::LLVM::LLVMPointerType::get(getPyLongObject());
            });

        for (auto op : module.getOps<pylir::Dialect::ConstantGlobalOp>())
        {
            // Type objects are ones that are of type `pylir.object<@__builtins__.type>`
            if (auto type = op.type().dyn_cast_or_null<pylir::Dialect::KnownTypeObjectType>();
                !type || type.getType().getValue() != llvm::StringRef{pylir::Dialect::typeTypeObjectName})
            {
                continue;
            }
            if (m_knownObjectTypes.count(op.sym_name()))
            {
                continue;
            }
            m_knownObjectTypes.insert({op.sym_name(), getTypeFromTypeObject(op)});
        }

        addConversion(
            [&](pylir::Dialect::KnownTypeObjectType type) -> llvm::Optional<mlir::Type>
            {
                auto value = type.getType().getValue();
                return mlir::LLVM::LLVMPointerType::get(m_knownObjectTypes.lookup(value));
            });
    }

    mlir::Type getPyObject()
    {
        if (!m_pyObject)
        {
            m_pyObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyObject");
            std::vector<mlir::Type> types;
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(getPyTypeObject())); // pointer to the type
            m_pyObject.setBody(types, false);
        }
        return m_pyObject;
    }

    mlir::Type getPyTupleObject()
    {
        if (!m_pyTupleObject)
        {
            m_pyTupleObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTupleObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the tuple for now
            m_pyTupleObject.setBody(types, false);
        }
        return m_pyTupleObject;
    }

    mlir::Type getPyDictObject()
    {
        if (!m_pyDictObject)
        {
            m_pyDictObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyDictObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the dict for now
            m_pyDictObject.setBody(types, false);
        }
        return m_pyDictObject;
    }

    mlir::Type getPyFunctionObject()
    {
        if (!m_pyFunctionObject)
        {
            m_pyFunctionObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyFunctionObject");
            std::vector<mlir::Type> types;
            auto pyObject = getPyObject();
            types.emplace_back(pyObject); // base
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::LLVM::LLVMFunctionType::get(pyObject, {pyObject, pyObject, pyObject})));
            m_pyFunctionObject.setBody(types, false);
        }
        return m_pyFunctionObject;
    }

    mlir::Type getPyLongObject()
    {
        if (!m_pyLongObject)
        {
            m_pyLongObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyLongObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the integer for now
            m_pyLongObject.setBody(types, false);
        }

        return m_pyLongObject;
    }

    mlir::Type getPyTypeObject()
    {
        if (!m_pyTypeObject)
        {
            m_pyTypeObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTypeObject");
            std::vector<mlir::Type> types;
            auto pyObject = getPyObject();
            types.emplace_back(pyObject); // base
            for (std::size_t i = 1; i <= pylir::Dialect::getMaxEnumValForTypeSlotPredicate(); i++)
            {
                auto pred = pylir::Dialect::symbolizeTypeSlotPredicate(i);
                PYLIR_ASSERT(pred);
                types.emplace_back(
                    convertType(pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(&getContext(), *pred)));
            }
            m_pyTypeObject.setBody(types, false);
        }
        return m_pyTypeObject;
    }
};

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

mlir::Value PyLong_FromSize_t(PylirTypeConverter& converter, mlir::OpBuilder& rewriter, mlir::Location location,
                              mlir::Value constant, mlir::ModuleOp module)
{
    auto func = declareFunc(mlir::LLVM::LLVMFunctionType::get(
                                mlir::LLVM::LLVMPointerType::get(converter.getPyLongObject()), {rewriter.getI64Type()}),
                            "PyLong_FromSize_t", module);
    return rewriter.create<mlir::LLVM::CallOp>(location, func, mlir::ValueRange{constant}).getResult(0);
}

template <class Self, class Op>
struct SingleOpMatcher : public mlir::ConvertToLLVMPattern
{
    using Operation = Op;

    SingleOpMatcher(llvm::StringRef rootOpName, mlir::MLIRContext* context, PylirTypeConverter& typeConverter,
                    mlir::PatternBenefit benefit = 1)
        : mlir::ConvertToLLVMPattern(rootOpName, context, typeConverter, benefit)
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

protected:
    PylirTypeConverter* getTypeConverter() const
    {
        return mlir::ConversionPattern::getTypeConverter<PylirTypeConverter>();
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
                    auto constant = createIndexConstant(rewriter, op->getLoc(), value.getValue().getSExtValue());

                    rewriter.replaceOp(op, {PyLong_FromSize_t(*getTypeConverter(), rewriter, op->getLoc(), constant,
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
                        op, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyDictObject()));
                })
            .Default([](auto) { PYLIR_UNREACHABLE; });
    }
};

struct ConstantGlobalOpConversion : SingleOpMatcher<ConstantGlobalOpConversion, pylir::Dialect::ConstantGlobalOp>
{
    std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors;

    ConstantGlobalOpConversion(std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors,
                               llvm::StringRef rootOpName, mlir::MLIRContext* context,
                               PylirTypeConverter& typeConverter, mlir::PatternBenefit benefit = 1)
        : SingleOpMatcher(rootOpName, context, typeConverter, benefit), ctors(ctors)
    {
    }

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ConstantGlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto exit = llvm::make_scope_exit([&] { rewriter.eraseOp(op); });
        auto resultType =
            typeConverter->convertType(adaptor.type().getValue()).dyn_cast_or_null<mlir::LLVM::LLVMPointerType>();
        PYLIR_ASSERT(resultType);
        auto newOp = rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType.getElementType(), false,
                                                           mlir::LLVM::Linkage::LinkonceODR,
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
            mlir::Value address = rewriter.create<mlir::LLVM::AddressOfOp>(op->getLoc(), newOp);
            address = rewriter.create<mlir::LLVM::BitcastOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()), address);
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                                rewriter.getI64IntegerAttr(0));
            auto typePointer = rewriter.create<mlir::LLVM::GEPOp>(
                op->getLoc(),
                mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject())),
                address, mlir::ValueRange{zero, zero});

            mlir::Value typeObject = rewriter.create<mlir::LLVM::AddressOfOp>(
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeObject()),
                adaptor.type().getValue().cast<pylir::Dialect::KnownTypeObjectType>().getType());
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), typePointer, typeObject);
            // TODO init

            rewriter.create<mlir::LLVM::ReturnOp>(op->getLoc(), mlir::ValueRange{});
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
        mlir::Type resultType = mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject());
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
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
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
                                                mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()),
                                                {/*TODO should be PyObject**/ mlir::LLVM::LLVMPointerType::get(
                                                     getTypeConverter()->getPyTupleObject()),
                                                 /*TODO should be size_t aka index*/ mlir::LLVM::LLVMPointerType::get(
                                                     getTypeConverter()->getPyLongObject())}),
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
        auto longPtr = mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyLongObject());
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
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()), toStore);
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
                op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()), input);
        }
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op->getResultTypes()[0])), input,
            mlir::ValueRange{zero, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
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
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
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
        rewriter.replaceOp(op, {PyLong_FromSize_t(*getTypeConverter(), rewriter, op->getLoc(), id,
                                                  op->getParentOfType<mlir::ModuleOp>())});
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
        auto iPtr = mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyLongObject());
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
        auto pyObjectStar = mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject());
        auto arrayType = mlir::LLVM::LLVMArrayType::get(pyObjectStar, adaptor.input().size());
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
        auto alloca =
            rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(arrayType), one, 8);
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        auto pyObjectStarStar = mlir::LLVM::LLVMPointerType::get(pyObjectStar);
        for (auto iter = adaptor.input().begin(); iter != adaptor.input().end(); iter++)
        {
            auto constant = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                                    rewriter.getI64IntegerAttr(iter.getIndex()));
            auto gep = rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), pyObjectStarStar, alloca,
                                                          mlir::ValueRange{zero, constant});
            rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), *iter, gep);
        }
        auto func = declareFunc(
            mlir::LLVM::LLVMFunctionType::get(typeConverter->convertType(rewriter.getType<pylir::Dialect::TupleType>()),
                                              {pyObjectStarStar, rewriter.getI64Type()}),
            "pylir_make_tuple", op->getParentOfType<mlir::ModuleOp>());
        auto size = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                            rewriter.getI64IntegerAttr(adaptor.input().size()));
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
            op, func,
            mlir::ValueRange{rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), pyObjectStarStar, alloca,
                                                                mlir::ValueRange{zero, zero}),
                             size});
    }
};

struct GetTypeSlotConversion : SingleOpMatcher<GetTypeSlotConversion, pylir::Dialect::GetTypeSlotOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetTypeSlotOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto zero = createIndexConstant(rewriter, op->getLoc(), 0);
        auto member = createIndexConstant(rewriter, op->getLoc(), adaptor.predicate().getInt());
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op->getResult(0).getType())),
            adaptor.input(), mlir::ValueRange{zero, member});
        auto loaded = rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), gep);
        mlir::Value found;
        if (loaded.getType().isa<mlir::LLVM::LLVMPointerType>())
        {
            auto null = rewriter.create<mlir::LLVM::NullOp>(op->getLoc(), loaded.getType());
            found = rewriter.create<mlir::LLVM::ICmpOp>(
                op->getLoc(), mlir::LLVM::ICmpPredicate::ne,
                rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), getIntPtrType(), loaded),
                rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), getIntPtrType(), null));
        }
        else
        {
            PYLIR_ASSERT(loaded.getType().isa<mlir::IntegerType>());
            auto null =
                rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), loaded.getType(), rewriter.getI64IntegerAttr(0));
            found = rewriter.create<mlir::LLVM::ICmpOp>(op->getLoc(), mlir::LLVM::ICmpPredicate::ne, loaded, null);
        }
        rewriter.replaceOp(op, {loaded, found});
    }
};

template <class... Args>
void populateSingleOpMatchers(PylirTypeConverter& converter, mlir::OwningRewritePatternList& patterns)
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
    PylirTypeConverter converter(module);

    populateSingleOpMatchers<GlobalOpConversion, ReinterpretOpConversion, ConstantOpConversion,
                             GetFunctionPointerConversion, GetItemConversion, IMulConversion, HandleOfConversion,
                             DataOfConversion, LoadConversion, StoreConversion, TypeOfConversion, AllocaConversion,
                             IdConversion, ICmpConversion, Bto1Conversion, MakeTupleConversion, GetTypeSlotConversion>(
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
    auto priority = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(),
                                                           builder.getI32IntegerAttr(65535));
    for (auto& [ctor, op] : ctors)
    {
        mlir::Value emptyStruct = builder.create<mlir::LLVM::UndefOp>(builder.getUnknownLoc(), literal);
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
