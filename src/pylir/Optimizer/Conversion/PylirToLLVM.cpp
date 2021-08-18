
#include "PylirToLLVM.hpp"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Dialect/PylirDialect.hpp>
#include <pylir/Optimizer/Dialect/PylirOps.hpp>
#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>
#include <pylir/Support/Macros.hpp>

#include <optional>

#include "PassDetail.hpp"
#include "WinX64.hpp"
#include "X86_64.hpp"

namespace
{
enum class RuntimeFunc
{
    PylirIntegerFromSizeT,
    PylirIntegerMul,
    PylirIntegerEqual,
    PylirIntegerNotEqual,
    PylirIntegerLess,
    PylirIntegerGreater,
    PylirIntegerLessOrEqual,
    PylirIntegerGreaterOrEqual,
    PylirGCAlloc,
    LAST_VALUE = PylirGCAlloc,
};

class PylirTypeConverter : public mlir::LLVMTypeConverter
{
    llvm::DenseMap<llvm::StringRef, mlir::Type> m_knownObjectTypes;
    mlir::LLVM::LLVMStructType m_pyObject;
    mlir::LLVM::LLVMStructType m_pyTupleValue;
    mlir::LLVM::LLVMStructType m_pyTupleObject;
    mlir::LLVM::LLVMStructType m_pyDictValue;
    mlir::LLVM::LLVMStructType m_pyDictObject;
    mlir::Type m_pyFunctionValue;
    mlir::LLVM::LLVMStructType m_pyFunctionObject;
    mlir::LLVM::LLVMStructType m_pyLongValue;
    mlir::LLVM::LLVMStructType m_pyLongObject;
    mlir::LLVM::LLVMStructType m_pyTypeObject;
    std::array<mlir::LLVM::LLVMFuncOp, static_cast<std::size_t>(RuntimeFunc::LAST_VALUE) + 1> m_runtimeFuncs;
    llvm::Triple m_triple;

    std::unique_ptr<pylir::Dialect::CABI> m_cAbi;

    mlir::Type getTypeFromTypeObject(pylir::Dialect::ConstantGlobalOp constant)
    {
        // builtin cases first
        auto value = constant.sym_name();
        if (value == llvm::StringRef{pylir::Dialect::typeTypeObjectName})
        {
            return getPyTypeObject();
        }
        if (value == llvm::StringRef{pylir::Dialect::intTypeObjectName})
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

    void initFunc(mlir::OpBuilder& builder, RuntimeFunc func)
    {
        mlir::Type returnType;
        llvm::SmallVector<mlir::Type> operands;
        std::string_view name;
        switch (func)
        {
            case RuntimeFunc::PylirIntegerFromSizeT:
                returnType = getPyLongValue();
                operands = {getIndexType()};
                name = "pylir_integer_from_size_t";
                break;
            case RuntimeFunc::PylirIntegerMul:
                returnType = getPyLongValue();
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_mul";
                break;
            case RuntimeFunc::PylirIntegerEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_equal";
                break;

            case RuntimeFunc::PylirIntegerNotEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_not_equal";
                break;

            case RuntimeFunc::PylirIntegerLess:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_less";
                break;

            case RuntimeFunc::PylirIntegerGreater:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_greater";
                break;

            case RuntimeFunc::PylirIntegerLessOrEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_less_or_equal";
                break;
            case RuntimeFunc::PylirIntegerGreaterOrEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {getPyLongValue(), getPyLongValue()};
                name = "pylir_integer_greater_or_equal";
                break;
            case RuntimeFunc::PylirGCAlloc:
                returnType = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(&getContext(), 8));
                operands = {getIndexType()};
                name = "pylir_gc_alloc";
                break;
            default: PYLIR_UNREACHABLE;
        }
        auto module = builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
        mlir::OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPointToEnd(module.getBody());
        m_runtimeFuncs[static_cast<std::size_t>(func)] =
            m_cAbi->declareFunc(builder, builder.getUnknownLoc(), returnType, name, operands);
    }

public:
    explicit PylirTypeConverter(mlir::ModuleOp module, llvm::StringRef dataLayout, llvm::StringRef triple)
        : mlir::LLVMTypeConverter(module.getContext(),
                                  [&dataLayout, &module]
                                  {
                                      mlir::LowerToLLVMOptions options(module.getContext());
                                      options.dataLayout = llvm::DataLayout{dataLayout};
                                      return options;
                                  }()),
          m_triple(triple)
    {
        addConversion(
            [&](pylir::Dialect::PointerType ptr) -> llvm::Optional<mlir::Type>
            {
                auto element = convertType(ptr.getElementType());
                if (!element)
                {
                    return llvm::None;
                }
                return mlir::LLVM::LLVMPointerType::get(element);
            });

        addConversion([&](pylir::Dialect::TupleType) -> llvm::Optional<mlir::Type> { return getPyTupleValue(); });

        addConversion([&](pylir::Dialect::DictType) -> llvm::Optional<mlir::Type> { return getPyDictValue(); });

        addConversion([&](pylir::Dialect::IntegerType) -> llvm::Optional<mlir::Type> { return getPyLongValue(); });

        addConversion([&](pylir::Dialect::TupleType) -> llvm::Optional<mlir::Type> { return getPyTupleValue(); });

        addConversion(
            [&](pylir::Dialect::ObjectType type) -> llvm::Optional<mlir::Type>
            {
                if (!type.getType())
                {
                    return getPyObject();
                }
                auto value = type.getType().getValue();
                return m_knownObjectTypes.lookup(value);
            });

        for (auto op : module.getOps<pylir::Dialect::ConstantGlobalOp>())
        {
            // Type objects are ones that are of type `pylir.object<@__builtins__.type>`
            if (auto type = op.type().dyn_cast_or_null<pylir::Dialect::ObjectType>();
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

        switch (m_triple.getArch())
        {
            case llvm::Triple::x86_64:
            {
                if (m_triple.isOSWindows())
                {
                    m_cAbi = std::make_unique<pylir::Dialect::WinX64>(mlir::DataLayout{module});
                }
                else
                {
                    m_cAbi = std::make_unique<pylir::Dialect::X86_64>(mlir::DataLayout{module});
                }
                break;
            }
            default:
                llvm::errs() << "ABI for target triple " << m_triple.str() << " has not been imlpemented yet. Sorry";
                std::terminate();
        }
    }

    mlir::Type getPyObject()
    {
        if (!m_pyObject)
        {
            m_pyObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyObject");
            std::vector<mlir::Type> types;
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(getPyTypeObject())); // pointer to the type
            auto result = m_pyObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyObject;
    }

    mlir::Type getPyTupleValue()
    {
        if (!m_pyTupleValue)
        {
            m_pyTupleValue = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTupleValue");
            std::vector<mlir::Type> types;
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the tuple for now
            auto result = m_pyTupleValue.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyTupleValue;
    }

    mlir::Type getPyTupleObject()
    {
        if (!m_pyTupleObject)
        {
            m_pyTupleObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTupleObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getPyTupleValue());
            auto result = m_pyTupleObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyTupleObject;
    }

    mlir::Type getPyDictValue()
    {
        if (!m_pyDictValue)
        {
            m_pyDictValue = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyDictValue");
            std::vector<mlir::Type> types;
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the dict for now
            auto result = m_pyDictValue.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyDictValue;
    }

    mlir::Type getPyDictObject()
    {
        if (!m_pyDictObject)
        {
            m_pyDictObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyDictObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getPyDictValue());
            auto result = m_pyDictObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyDictObject;
    }

    mlir::Type getPyFunctionValue()
    {
        if (!m_pyFunctionValue)
        {
            m_pyFunctionValue = convertType(pylir::Dialect::getCCFuncType(&getContext()));
        }
        return m_pyFunctionValue;
    }

    mlir::Type getPyFunctionObject()
    {
        if (!m_pyFunctionObject)
        {
            m_pyFunctionObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyFunctionObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getPyFunctionValue());
            auto result = m_pyFunctionObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyFunctionObject;
    }

    mlir::Type getPyLongValue()
    {
        if (!m_pyLongValue)
        {
            m_pyLongValue = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyLongValue");
            std::vector<mlir::Type> types;
            types.emplace_back(mlir::LLVM::LLVMPointerType::get(
                mlir::IntegerType::get(&getContext(), 8))); // opaque pointer to the integer for now
            auto result = m_pyLongValue.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }

        return m_pyLongValue;
    }

    mlir::Type getPyLongObject()
    {
        if (!m_pyLongObject)
        {
            m_pyLongObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyLongObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getPyLongValue());
            auto result = m_pyLongObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
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
            auto result = m_pyTypeObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyTypeObject;
    }

    mlir::Value callRuntime(mlir::OpBuilder& builder, mlir::Location loc, RuntimeFunc func, mlir::ValueRange operands)
    {
        auto index = static_cast<std::size_t>(func);
        if (!m_runtimeFuncs[index])
        {
            initFunc(builder, func);
        }
        return m_cAbi->callFunc(builder, loc, m_runtimeFuncs[index], operands);
    }
};

mlir::Value genNullConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type llvmType)
{
    return llvm::TypeSwitch<mlir::Type, mlir::Value>(llvmType)
        .Case<mlir::LLVM::LLVMPointerType>([&](auto ptrType)
                                           { return builder.create<mlir::LLVM::NullOp>(loc, ptrType); })
        .Case<mlir::IntegerType>(
            [&](auto integerType) {
                return builder.create<mlir::LLVM::ConstantOp>(loc, integerType, builder.getIntegerAttr(integerType, 0));
            })
        .Case<mlir::FloatType>(
            [&](auto floatType)
            { return builder.create<mlir::LLVM::ConstantOp>(loc, floatType, builder.getFloatAttr(floatType, 0)); })
        .Case<mlir::LLVM::LLVMStructType>(
            [&](mlir::LLVM::LLVMStructType structType)
            {
                mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(loc, structType);
                for (std::size_t i = 0; i < structType.getBody().size(); i++)
                {
                    undef = builder.create<mlir::LLVM::InsertValueOp>(
                        loc, undef, genNullConstant(builder, loc, structType.getBody()[i]),
                        builder.getI64ArrayAttr({static_cast<std::int64_t>(i)}));
                }
                return undef;
            })
        .Default([](auto) -> mlir::Value { PYLIR_UNREACHABLE; });
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

    mlir::Value normalizeObject(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value object) const
    {
        if (auto pointer = object.getType().dyn_cast_or_null<mlir::LLVM::LLVMPointerType>();
            pointer && pointer.getElementType() != getTypeConverter()->getPyObject())
        {
            return builder.create<mlir::LLVM::BitcastOp>(
                loc, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()), object);
        }
        return object;
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
                    rewriter.replaceOp(op, {getTypeConverter()->callRuntime(
                                               rewriter, op->getLoc(), RuntimeFunc::PylirIntegerFromSizeT, constant)});
                })
            .Default([](auto) { PYLIR_UNREACHABLE; });
    }
};

template <class T, class Op>
struct GlobalConversionBase : SingleOpMatcher<T, Op>
{
    std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors;

    GlobalConversionBase(std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>>& ctors,
                         llvm::StringRef rootOpName, mlir::MLIRContext* context, PylirTypeConverter& typeConverter,
                         mlir::PatternBenefit benefit = 1)
        : SingleOpMatcher<T, Op>(rootOpName, context, typeConverter, benefit), ctors(ctors)
    {
    }

    void genInitializer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, mlir::LLVM::GlobalOp newOp,
                        mlir::Attribute initAttr, llvm::StringRef typeName, llvm::StringRef symName) const
    {
        llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> dictInit;

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            auto* block = new mlir::Block;
            newOp.getInitializerRegion().push_back(block);
            rewriter.setInsertionPointToStart(block);
            mlir::Value initializer = genNullConstant(rewriter, loc, newOp.getType());
            if (typeName == llvm::StringRef{pylir::Dialect::typeTypeObjectName})
            {
                llvm::DenseSet<pylir::Dialect::TypeSlotPredicate> seen;
                for (auto [name, value] : initAttr.cast<pylir::Dialect::DictAttr>().getValue())
                {
                    auto string = name.template dyn_cast<mlir::StringAttr>();
                    if (!string)
                    {
                        dictInit.emplace_back(name, value);
                        continue;
                    }
                    auto known = pylir::Dialect::symbolizeTypeSlotPredicate(string.getValue());
                    if (!known)
                    {
                        dictInit.emplace_back(name, value);
                        continue;
                    }
                    seen.insert(*known);
                    mlir::Value constant = llvm::TypeSwitch<mlir::Attribute, mlir::Value>(value)
                                               .Case<mlir::IntegerAttr>(
                                                   [&](mlir::IntegerAttr attr) {
                                                       return rewriter.create<mlir::LLVM::ConstantOp>(
                                                           loc, this->typeConverter->convertType(attr.getType()), attr);
                                                   })
                                               .template Case<mlir::FlatSymbolRefAttr>(
                                                   [&](mlir::FlatSymbolRefAttr attr)
                                                   {
                                                       return rewriter.create<mlir::LLVM::AddressOfOp>(
                                                           loc,
                                                           this->typeConverter->convertType(
                                                               pylir::Dialect::GetTypeSlotOp::returnTypeFromPredicate(
                                                                   this->getContext(), *known)),
                                                           attr);
                                                   })
                                               .Default([](auto) -> mlir::Value { PYLIR_UNREACHABLE; });
                    initializer = rewriter.create<mlir::LLVM::InsertValueOp>(
                        loc, initializer, constant, rewriter.getI64ArrayAttr({static_cast<std::int64_t>(*known)}));
                }
            }
            else if (typeName == llvm::StringRef{pylir::Dialect::noneTypeObjectName}
                     || typeName == llvm::StringRef{pylir::Dialect::notImplementedTypeObjectName})
            {
            }
            if (!typeName.empty())
            {
                auto constant = rewriter.create<mlir::LLVM::AddressOfOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyTypeObject()),
                    rewriter.getSymbolRefAttr(typeName));

                llvm::SmallVector<std::int64_t> indices{0};
                if (newOp.getType() != this->getTypeConverter()->getPyObject())
                {
                    indices.push_back(0);
                }
                initializer = rewriter.create<mlir::LLVM::InsertValueOp>(loc, initializer, constant,
                                                                         rewriter.getI64ArrayAttr(indices));
            }
            rewriter.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{initializer});
        }

        if (dictInit.empty())
        {
            return;
        }

        mlir::LLVM::LLVMFuncOp ctor;

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            ctor = rewriter.create<mlir::LLVM::LLVMFuncOp>(
                loc, (symName + ".ctor").str(),
                mlir::LLVM::LLVMFunctionType::get(rewriter.getType<mlir::LLVM::LLVMVoidType>(), {}), newOp.linkage());
            rewriter.setInsertionPointToStart(ctor.addEntryBlock());
            mlir::Value address = rewriter.create<mlir::LLVM::AddressOfOp>(loc, newOp);
            address = rewriter.create<mlir::LLVM::BitcastOp>(
                loc, mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyObject()), address);
            auto zero =
                rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
            auto typePointer =
                rewriter.create<mlir::LLVM::GEPOp>(loc,
                                                   mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(
                                                       this->getTypeConverter()->getPyObject())),
                                                   address, mlir::ValueRange{zero, zero});

            mlir::Value typeObject = rewriter.create<mlir::LLVM::AddressOfOp>(
                loc, mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyTypeObject()), typeName);
            rewriter.create<mlir::LLVM::StoreOp>(loc, typePointer, typeObject);
            // TODO init

            rewriter.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
        }

        ctors.emplace_back(ctor, newOp);
    }
};

struct ConstantGlobalOpConversion : GlobalConversionBase<ConstantGlobalOpConversion, pylir::Dialect::ConstantGlobalOp>
{
    using GlobalConversionBase::GlobalConversionBase;

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
        genInitializer(rewriter, op->getLoc(), newOp, adaptor.initializer(),
                       adaptor.type().getValue().cast<pylir::Dialect::ObjectType>().getType().getValue(),
                       adaptor.sym_name().getValue());
    }
};

struct GlobalOpConversion : GlobalConversionBase<GlobalOpConversion, pylir::Dialect::GlobalOp>
{
    using GlobalConversionBase::GlobalConversionBase;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto exit = llvm::make_scope_exit([&] { rewriter.eraseOp(op); });
        auto resultType = typeConverter->convertType(adaptor.type().getValue());
        PYLIR_ASSERT(resultType);
        auto newOp =
            rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType, false, mlir::LLVM::Linkage::LinkonceODR,
                                                  adaptor.sym_name().getValue(), mlir::Attribute{});
        if (adaptor.initial_value())
        {
            llvm::StringRef typeName;
            if (auto object = adaptor.type().getValue().dyn_cast<pylir::Dialect::ObjectType>())
            {
                typeName = object.getType().getValue();
            }
            genInitializer(rewriter, op->getLoc(), newOp, adaptor.initial_value(), typeName,
                           adaptor.sym_name().getValue());
        }
    }
};

struct GetGlobalConversion : SingleOpMatcher<GetGlobalConversion, pylir::Dialect::GetGlobalOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetGlobalOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                             adaptor.name());
    }
};

struct BoxConversion : SingleOpMatcher<BoxConversion, pylir::Dialect::BoxOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::BoxOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto resultType = typeConverter->convertType(op->getResultTypes()[0]);
        mlir::Value undef = rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), resultType);
        auto typeObject = rewriter.create<mlir::LLVM::AddressOfOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeObject()),
            op->getResultTypes()[0].cast<pylir::Dialect::ObjectType>().getType());
        undef = rewriter.create<mlir::LLVM::InsertValueOp>(op->getLoc(), undef, typeObject,
                                                           rewriter.getI64ArrayAttr({0, 0}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(op, undef, adaptor.input(),
                                                               rewriter.getI64ArrayAttr({1}));
    }
};

struct UnboxConversion : SingleOpMatcher<UnboxConversion, pylir::Dialect::UnboxOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::UnboxOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto resultType = typeConverter->convertType(op->getResultTypes()[0]);
        // TODO: add other builtins
        auto objectType = llvm::TypeSwitch<mlir::Type, mlir::Type>(op->getResultTypes()[0])
                              .Case([&](pylir::Dialect::IntegerType) { return getTypeConverter()->getPyLongObject(); })
                              .Case([&](pylir::Dialect::TupleType) { return getTypeConverter()->getPyTypeObject(); })
                              .Case([&](mlir::FunctionType) { return getTypeConverter()->getPyFunctionObject(); })
                              .Default([](auto) -> mlir::Type { PYLIR_UNREACHABLE; });
        auto cast = rewriter.create<mlir::LLVM::BitcastOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(objectType),
                                                           adaptor.input());
        auto zero = createIndexConstant(rewriter, op->getLoc(), 0);
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(resultType), cast,
                                                      mlir::ValueRange{zero, one});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
    }
};

struct GCAllocConversion : SingleOpMatcher<GCAllocConversion, pylir::Dialect::GCAllocOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GCAllocOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto type = typeConverter->convertType(op->getResultTypes()[0]);
        auto null = rewriter.create<mlir::LLVM::NullOp>(op->getLoc(), type);
        auto one = createIndexConstant(rewriter, op->getLoc(), 1);
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), type, null, mlir::ValueRange{one});
        mlir::Value sizeOf = rewriter.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), getIndexType(), gep);
        if (adaptor.dynamicSize())
        {
            sizeOf = rewriter.create<mlir::LLVM::MulOp>(op->getLoc(), sizeOf, adaptor.dynamicSize());
        }
        auto alloc = getTypeConverter()->callRuntime(rewriter, op->getLoc(), RuntimeFunc::PylirGCAlloc, {sizeOf});
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, type, alloc);
    }
};

struct AllocaConversion : SingleOpMatcher<AllocaConversion, pylir::Dialect::AllocaOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::AllocaOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto type = typeConverter->convertType(op->getResultTypes()[0]);
        mlir::Value sizeOf;
        if (adaptor.dynamicSize())
        {
            sizeOf = adaptor.dynamicSize();
        }
        else
        {
            sizeOf = createIndexConstant(rewriter, op->getLoc(), 1);
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(op, type, sizeOf);
    }
};

struct CallConversion : SingleOpMatcher<CallConversion, pylir::Dialect::CallOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::CallOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        llvm::SmallVector<mlir::Value> operands;
        for (auto arg : adaptor.operands())
        {
            operands.push_back(normalizeObject(rewriter, op->getLoc(), arg));
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                        adaptor.callee(), operands);
    }
};

struct CallIndirectConversion : SingleOpMatcher<CallIndirectConversion, pylir::Dialect::CallIndirectOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::CallIndirectOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        llvm::SmallVector<mlir::Value> operands{adaptor.callee()};
        for (auto arg : adaptor.operands())
        {
            operands.push_back(normalizeObject(rewriter, op->getLoc(), arg));
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                        operands);
    }
};

struct ReturnConversion : SingleOpMatcher<ReturnConversion, pylir::Dialect::ReturnOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ReturnOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        llvm::SmallVector<mlir::Value> operands;
        for (auto arg : adaptor.operands())
        {
            operands.push_back(normalizeObject(rewriter, op->getLoc(), arg));
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, operands);
    }
};

struct IMulConversion : SingleOpMatcher<IMulConversion, pylir::Dialect::IMulOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::IMulOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOp(op, getTypeConverter()->callRuntime(rewriter, op->getLoc(), RuntimeFunc::PylirIntegerMul,
                                                               {adaptor.lhs(), adaptor.rhs()}));
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, adaptor.pointer());
    }
};

struct StoreConversion : SingleOpMatcher<StoreConversion, pylir::Dialect::StoreOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::StoreOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto toStore = normalizeObject(rewriter, op->getLoc(), adaptor.value());
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, toStore, adaptor.pointer());
    }
};

struct TypeOfConversion : SingleOpMatcher<TypeOfConversion, pylir::Dialect::TypeOfOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::TypeOfOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        llvm::SmallVector<std::int64_t> indices{0};
        // One 0 as the very first field is the pointer to the type object
        // The second if it is not a PyObject, but a derived type that we need to navigate to
        if (adaptor.input().getType() != getTypeConverter()->getPyObject())
        {
            indices.push_back(0);
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, typeConverter->convertType(op->getResultTypes()[0]),
                                                                adaptor.input(), rewriter.getI64ArrayAttr(indices));
    }
};

struct IdConversion : SingleOpMatcher<IdConversion, pylir::Dialect::IdOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::IdOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, getIndexType(), adaptor.input());
    }
};

struct ICmpConversion : SingleOpMatcher<ICmpConversion, pylir::Dialect::ICmpOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ICmpOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        RuntimeFunc func;
        switch (mlir::cast<pylir::Dialect::ICmpOp>(op).predicate())
        {
            case pylir::Dialect::CmpPredicate::EQ: func = RuntimeFunc::PylirIntegerEqual; break;
            case pylir::Dialect::CmpPredicate::NE: func = RuntimeFunc::PylirIntegerNotEqual; break;
            case pylir::Dialect::CmpPredicate::LT: func = RuntimeFunc::PylirIntegerLess; break;
            case pylir::Dialect::CmpPredicate::LE: func = RuntimeFunc::PylirIntegerLessOrEqual; break;
            case pylir::Dialect::CmpPredicate::GT: func = RuntimeFunc::PylirIntegerGreater; break;
            case pylir::Dialect::CmpPredicate::GE: func = RuntimeFunc::PylirIntegerGreaterOrEqual; break;
            default: PYLIR_UNREACHABLE;
        }
        rewriter.replaceOp(
            op, getTypeConverter()->callRuntime(rewriter, op->getLoc(), func, {adaptor.lhs(), adaptor.rhs()}));
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
        auto member = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                                              rewriter.getI32IntegerAttr(adaptor.predicate().getInt()));
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

    std::vector<mlir::SymbolOpInterface> linkOnceODR;
    for (auto iter : module.getOps<mlir::SymbolOpInterface>())
    {
        if (iter->hasAttr("linkonce"))
        {
            linkOnceODR.emplace_back(iter);
        }
    }

    mlir::RewritePatternSet patterns(&getContext());
    PylirTypeConverter converter(module, dataLayout.getValue(), targetTriple.getValue());

    populateSingleOpMatchers<ConstantOpConversion, BoxConversion, IMulConversion, DataOfConversion, StoreConversion,
                             TypeOfConversion, IdConversion, ICmpConversion, GetTypeSlotConversion, GCAllocConversion,
                             LoadConversion, GetGlobalConversion, AllocaConversion, CallConversion,
                             CallIndirectConversion, UnboxConversion, ReturnConversion>(converter, patterns);
    std::vector<std::pair<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>> ctors;
    patterns.insert<ConstantGlobalOpConversion>(ctors, ConstantGlobalOpConversion::Operation::getOperationName(),
                                                &converter.getContext(), converter);
    patterns.insert<GlobalOpConversion>(ctors, GlobalOpConversion::Operation::getOperationName(),
                                        &converter.getContext(), converter);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);

    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(module.getContext(), targetTriple.getValue()));
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(module.getContext(), dataLayout.getValue()));

    mlir::LLVMConversionTarget target(getContext());
    target.addIllegalDialect<pylir::Dialect::PylirDialect, mlir::StandardOpsDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
        return;
    }

    auto table = mlir::SymbolTable(module);
    for (auto& iter : linkOnceODR)
    {
        auto op = table.lookup(iter.getName());
        PYLIR_ASSERT(op);
        llvm::TypeSwitch<mlir::Operation*>(op)
            .Case<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>(
                [](auto op)
                { op.linkageAttr(mlir::LLVM::LinkageAttr::get(op.getContext(), mlir::LLVM::Linkage::LinkonceODR)); })
            .Default([](auto) { PYLIR_UNREACHABLE; });
    }

    if (ctors.empty())
    {
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
