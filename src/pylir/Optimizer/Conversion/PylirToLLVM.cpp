
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
    PylirIntegerToIndex,
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
    mlir::LLVM::LLVMStructType m_pyTupleObject;
    mlir::LLVM::LLVMStructType m_pyDictValue;
    mlir::LLVM::LLVMStructType m_pyDictObject;
    mlir::Type m_pyFunctionValue;
    mlir::LLVM::LLVMStructType m_pyFunctionObject;
    mlir::LLVM::LLVMStructType m_pyLongValue;
    mlir::LLVM::LLVMStructType m_pyIntObject;
    mlir::LLVM::LLVMStructType m_pyTypeObject;
    std::array<mlir::LLVM::LLVMFuncOp, static_cast<std::size_t>(RuntimeFunc::LAST_VALUE) + 1> m_runtimeFuncs;
    llvm::Triple m_triple;

    std::unique_ptr<pylir::Dialect::CABI> m_cAbi;

    mlir::Type getTypeFromTypeObject(pylir::Dialect::ConstantGlobalOp constant)
    {
        auto value = constant.sym_name();
        if (value == llvm::StringRef{pylir::Dialect::typeTypeObjectName})
        {
            return getPyTypeObject();
        }
        if (value == llvm::StringRef{pylir::Dialect::tupleTypeObjectName})
        {
            return getPyTupleObject();
        }
        if (value == llvm::StringRef{pylir::Dialect::intTypeObjectName})
        {
            return getPyIntObject();
        }
        // TODO custom classes
        return getPyObject();
    }

    void initFunc(mlir::OpBuilder& builder, RuntimeFunc func)
    {
        mlir::Type returnType;
        llvm::SmallVector<mlir::Type> operands;
        llvm::SmallVector<mlir::Attribute> passthrough;
        std::string_view name;
        auto pyObjectRef = mlir::LLVM::LLVMPointerType::get(getPyObject());
        switch (func)
        {
            case RuntimeFunc::PylirIntegerFromSizeT:
                returnType = pyObjectRef;
                operands = {getIndexType()};
                name = "pylir_integer_from_size_t";
                break;
            case RuntimeFunc::PylirIntegerToIndex:
                returnType =
                    mlir::LLVM::LLVMStructType::getLiteral(&getContext(), {getIndexType(), builder.getI1Type()});
                operands = {pyObjectRef};
                name = "pylir_integer_to_index";
                break;
            case RuntimeFunc::PylirIntegerMul:
                returnType = pyObjectRef;
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_mul";
                break;
            case RuntimeFunc::PylirIntegerEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_equal";
                break;

            case RuntimeFunc::PylirIntegerNotEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_not_equal";
                break;

            case RuntimeFunc::PylirIntegerLess:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_less";
                break;

            case RuntimeFunc::PylirIntegerGreater:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_greater";
                break;

            case RuntimeFunc::PylirIntegerLessOrEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
                name = "pylir_integer_less_or_equal";
                break;
            case RuntimeFunc::PylirIntegerGreaterOrEqual:
                returnType = mlir::IntegerType::get(&getContext(), 1);
                operands = {pyObjectRef, pyObjectRef};
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
        if (!passthrough.empty())
        {
            m_runtimeFuncs[static_cast<std::size_t>(func)]->setAttr("passthrough",
                                                                    mlir::ArrayAttr::get(&getContext(), passthrough));
        }
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

        addConversion([&](pylir::Dialect::ObjectType) -> llvm::Optional<mlir::Type> { return getPyObject(); });

        for (auto op : module.getOps<pylir::Dialect::ConstantGlobalOp>())
        {
            // Type objects are ones that are of type `pylir.object<@__builtins__.type>`
            if (op.type() != llvm::StringRef{pylir::Dialect::typeTypeObjectName})
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

    mlir::Type getPyTupleObject()
    {
        if (!m_pyTupleObject)
        {
            m_pyTupleObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTupleObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getIndexType());
            types.emplace_back(mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getPyObject()), 0));
            auto result = m_pyTupleObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyTupleObject;
    }

    mlir::Type getPyIntObject()
    {
        if (!m_pyIntObject)
        {
            m_pyIntObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyIntObject");
            std::vector<mlir::Type> types;
            types.emplace_back(getPyObject()); // base
            types.emplace_back(getIndexType());
            types.emplace_back(mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getIndexType()), 0));
            auto result = m_pyIntObject.setBody(types, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return m_pyIntObject;
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

    mlir::Type getTypeFromTypeObject(llvm::StringRef ref)
    {
        return m_knownObjectTypes.lookup(ref);
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
        auto resultType = getTypeConverter()->getTypeFromTypeObject(adaptor.type().getValue());
        PYLIR_ASSERT(resultType);
        auto newOp =
            rewriter.create<mlir::LLVM::GlobalOp>(op->getLoc(), resultType, false, mlir::LLVM::Linkage::LinkonceODR,
                                                  adaptor.sym_name().getValue(), mlir::Attribute{});
        genInitializer(rewriter, op->getLoc(), newOp, adaptor.initializer(), adaptor.type().getValue(),
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
            genInitializer(rewriter, op->getLoc(), newOp, adaptor.initial_value(), {}, adaptor.sym_name().getValue());
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

struct BoxConversion : SingleOpMatcher<BoxConversion, pylir::Dialect::BoxIntoOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::BoxIntoOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto exit = llvm::make_scope_exit([&] { rewriter.eraseOp(op); });
        // TODO
        PYLIR_UNREACHABLE;
    }
};

struct UnboxConversion : SingleOpMatcher<UnboxConversion, pylir::Dialect::UnboxOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::UnboxOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        // TODO
        PYLIR_UNREACHABLE;
    }
};

template <class Sub, class Op>
struct BaseObjectAllocConversion : SingleOpMatcher<Sub, Op>
{
    using SingleOpMatcher<Sub, Op>::SingleOpMatcher;

    using SingleOpMatcher<Sub, Op>::rewrite;

    void rewrite(mlir::Operation* op, typename Op::Adaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const
    {
        auto type = this->getTypeConverter()->getTypeFromTypeObject(adaptor.type().getValue());
        auto loc = op->getLoc();
        mlir::Value size = this->getSizeInBytes(loc, type, rewriter);
        if (adaptor.variableSize())
        {
            auto elementType = type.template cast<mlir::LLVM::LLVMStructType>()
                                   .getBody()
                                   .back()
                                   .template cast<mlir::LLVM::LLVMArrayType>()
                                   .getElementType();
            auto elementSize = this->getSizeInBytes(loc, elementType, rewriter);
            auto appendage = rewriter.create<mlir::LLVM::MulOp>(loc, adaptor.variableSize(), elementSize);
            size = rewriter.create<mlir::LLVM::AddOp>(loc, size, appendage);
        }

        auto alloc = static_cast<const Sub*>(this)->doAllocation(rewriter, op, size);

        auto object = rewriter.create<mlir::LLVM::BitcastOp>(loc, mlir::LLVM::LLVMPointerType::get(type), alloc);

        auto zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto typeGEP =
            rewriter.create<mlir::LLVM::GEPOp>(loc,
                                               mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(
                                                   this->getTypeConverter()->getPyTypeObject())),
                                               object, mlir::ValueRange{zero, zero, zero});
        auto typeRef = rewriter.create<mlir::LLVM::AddressOfOp>(
            loc, mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyTypeObject()), adaptor.type());
        rewriter.create<mlir::LLVM::StoreOp>(loc, typeRef, typeGEP);

        if (adaptor.variableSize())
        {
            auto one =
                rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
            auto countGEP = rewriter.create<mlir::LLVM::GEPOp>(
                loc, mlir::LLVM::LLVMPointerType::get(this->getIndexType()), object, mlir::ValueRange{zero, one});
            rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.variableSize(), countGEP);
        }
    }
};

struct GCObjectAllocConversion : BaseObjectAllocConversion<GCObjectAllocConversion, pylir::Dialect::GCObjectAllocOp>
{
    using BaseObjectAllocConversion::BaseObjectAllocConversion;

    using SingleOpMatcher::rewrite;

    using BaseObjectAllocConversion::rewrite;

    mlir::Value doAllocation(mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op, mlir::Value size) const
    {
        auto alloc = this->getTypeConverter()->callRuntime(rewriter, op->getLoc(), RuntimeFunc::PylirGCAlloc, {size});
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
            op, this->typeConverter->convertType(op->getResultTypes()[0]), alloc);
        return alloc;
    }
};

struct ObjectAllocaConversion : BaseObjectAllocConversion<ObjectAllocaConversion, pylir::Dialect::ObjectAllocaOp>
{
    using BaseObjectAllocConversion::BaseObjectAllocConversion;

    using SingleOpMatcher::rewrite;

    using BaseObjectAllocConversion::rewrite;

    mlir::Value doAllocation(mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op, mlir::Value size) const
    {
        auto alloc = rewriter.create<mlir::LLVM::AllocaOp>(op->getLoc(), rewriter.getIntegerType(8), size, /*TODO*/ 8);
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
            op, this->typeConverter->convertType(op->getResultTypes()[0]), alloc);
        return alloc;
    }
};

struct AllocaConversion : SingleOpMatcher<AllocaConversion, pylir::Dialect::AllocaOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::AllocaOp::Adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto type = typeConverter->convertType(op->getResultTypes()[0]);
        rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(op, type, createIndexConstant(rewriter, op->getLoc(), 1));
    }
};

struct IntegerConstantConversion : SingleOpMatcher<IntegerConstantConversion, pylir::Dialect::IntegerConstant>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::IntegerConstant::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        // TODO strings instead of size_t
        auto value = adaptor.value().getValue().getSExtValue();
        auto result = getTypeConverter()->callRuntime(rewriter, op->getLoc(), RuntimeFunc::PylirIntegerFromSizeT,
                                                      {createIndexConstant(rewriter, op->getLoc(), value)});
        rewriter.replaceOp(op, result);
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

struct ItoIndexConversion : SingleOpMatcher<ItoIndexConversion, pylir::Dialect::ItoIndexOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::ItoIndexOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto result = getTypeConverter()->callRuntime(rewriter, op->getLoc(), RuntimeFunc::PylirIntegerToIndex,
                                                      {adaptor.input()});
        auto first = rewriter.create<mlir::LLVM::ExtractValueOp>(op->getLoc(), getIndexType(), result,
                                                                 rewriter.getI32ArrayAttr({0}));
        auto second = rewriter.create<mlir::LLVM::ExtractValueOp>(op->getLoc(), rewriter.getI1Type(), result,
                                                                  rewriter.getI32ArrayAttr({1}));
        rewriter.replaceOp(op, {first, second});
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), adaptor.pointer());
    }
};

struct TypeOfConversion : SingleOpMatcher<TypeOfConversion, pylir::Dialect::TypeOfOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::TypeOfOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeObject())),
            adaptor.input(), mlir::ValueRange{zero, zero});
        auto typeObject = rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), gep);
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
            op, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject()), typeObject);
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
        auto zero =
            rewriter.create<mlir::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto member = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                                              rewriter.getI32IntegerAttr(adaptor.predicate().getInt()));

        auto typeObject = rewriter.create<mlir::LLVM::BitcastOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeObject()), adaptor.input());

        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op->getResult(0).getType())),
            typeObject, mlir::ValueRange{zero, member});
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

struct GetTupleItemConversion : SingleOpMatcher<GetTupleItemConversion, pylir::Dialect::GetTupleItemOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::GetTupleItemOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto tupleObject = rewriter.create<mlir::LLVM::BitcastOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTupleObject()), adaptor.tuple());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject())),
            tupleObject, mlir::ValueRange{zero, two, adaptor.index()});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
    }
};

struct SetTupleItemConversion : SingleOpMatcher<SetTupleItemConversion, pylir::Dialect::SetTupleItemOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::SetTupleItemOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto tupleObject = rewriter.create<mlir::LLVM::BitcastOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTupleObject()), adaptor.memory());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op->getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObject())),
            tupleObject, mlir::ValueRange{zero, two, adaptor.index()});
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.element(), gep);
    }
};

struct TupleSizeConversion : SingleOpMatcher<TupleSizeConversion, pylir::Dialect::TupleSizeOp>
{
    using SingleOpMatcher::SingleOpMatcher;

    using SingleOpMatcher::rewrite;

    void rewrite(mlir::Operation* op, pylir::Dialect::TupleSizeOp::Adaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const
    {
        auto tupleObject = rewriter.create<mlir::LLVM::BitcastOp>(
            op->getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTupleObject()), adaptor.tuple());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op->getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      tupleObject, mlir::ValueRange{zero, one});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
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

    std::vector<std::string> linkOnceODR;
    for (auto iter : module.getOps<mlir::SymbolOpInterface>())
    {
        if (iter->hasAttr("linkonce"))
        {
            linkOnceODR.emplace_back(iter.getName());
        }
    }

    mlir::RewritePatternSet patterns(&getContext());
    PylirTypeConverter converter(module, dataLayout.getValue(), targetTriple.getValue());

    populateSingleOpMatchers<BoxConversion, IMulConversion, DataOfConversion, StoreConversion, TypeOfConversion,
                             IdConversion, ICmpConversion, GetTypeSlotConversion, GCObjectAllocConversion,
                             ObjectAllocaConversion, LoadConversion, GetGlobalConversion, AllocaConversion,
                             UnboxConversion, GetTupleItemConversion, SetTupleItemConversion, IntegerConstantConversion,
                             ItoIndexConversion, TupleSizeConversion>(converter, patterns);
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
        auto op = table.lookup(iter);
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
