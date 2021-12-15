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
    llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> m_globalBuffers;
    mlir::SymbolTable m_symbolTable;
    std::unique_ptr<pylir::CABI> m_cabi;

    mlir::LLVM::LLVMArrayType getSlotEpilogue(unsigned slotSize = 0)
    {
        return mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType()), slotSize);
    }

    mlir::LLVM::LLVMStructType getBufferComponent(mlir::Type elementType)
    {
        // TODO: should this be 3 pointers ala std::vector?
        return mlir::LLVM::LLVMStructType::getLiteral(
            &getContext(), {getIndexType(), getIndexType(), mlir::LLVM::LLVMPointerType::get(elementType)});
    }

public:
    PylirTypeConverter(mlir::MLIRContext* context, llvm::Triple triple, llvm::DataLayout dataLayout,
                       mlir::ModuleOp moduleOp)
        : mlir::LLVMTypeConverter(context,
                                  [&]
                                  {
                                      mlir::LowerToLLVMOptions options(context);
                                      options.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::None;
                                      options.dataLayout = dataLayout;
                                      return options;
                                  }()),
          m_symbolTable(moduleOp)
    {
        switch (triple.getArch())
        {
            case llvm::Triple::x86_64:
            {
                if (triple.isOSWindows())
                {
                    m_cabi = std::make_unique<pylir::WinX64>(mlir::DataLayout{moduleOp});
                }
                else
                {
                    m_cabi = std::make_unique<pylir::X86_64>(mlir::DataLayout{moduleOp});
                }
                break;
            }
            default: llvm::errs() << triple.str() << " not yet implemented"; std::abort();
        }
    }

    mlir::LLVM::LLVMStructType getPyObjectType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getSlotEpilogue(*slotSize)});
        }
        auto pyObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyObject");
        if (!pyObject.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyObject.setBody({mlir::LLVM::LLVMPointerType::get(pyObject),
                                  mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(pyObject), 0)},
                                 false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyObject;
    }

    mlir::LLVM::LLVMStructType getPyFunctionType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                convertType(pylir::Py::getUniversalCCType(&getContext())), getSlotEpilogue(*slotSize)});
        }
        auto pyFunction = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyFunction");
        if (!pyFunction.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyFunction.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                    convertType(pylir::Py::getUniversalCCType(&getContext())), getSlotEpilogue()},
                                   false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyFunction;
    }

    mlir::LLVM::LLVMStructType getPySequenceType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                 getBufferComponent(mlir::LLVM::LLVMPointerType::get(getPyObjectType())), getSlotEpilogue(*slotSize)});
        }
        auto pySequence = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PySequence");
        if (!pySequence.isInitialized())
        {
            [[maybe_unused]] auto result = pySequence.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                 getBufferComponent(mlir::LLVM::LLVMPointerType::get(getPyObjectType())), getSlotEpilogue()},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pySequence;
    }

    mlir::LLVM::LLVMStructType getPyStringType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(&getContext(),
                                                          {mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                                           getBufferComponent(mlir::IntegerType::get(&getContext(), 8)),
                                                           getSlotEpilogue(*slotSize)});
        }
        auto pyString = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyString");
        if (!pyString.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyString.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                  getBufferComponent(mlir::IntegerType::get(&getContext(), 8)), getSlotEpilogue()},
                                 false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyString;
    }

    mlir::LLVM::LLVMStructType getPyTypeType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getIndexType(), getSlotEpilogue(*slotSize)});
        }
        auto pyType = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyType");
        if (!pyType.isInitialized())
        {
            [[maybe_unused]] auto result = pyType.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getIndexType(), getSlotEpilogue()}, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyType;
    }

    mlir::LLVM::LLVMStructType getInstanceType(llvm::StringRef builtinsName)
    {
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Tuple.name}
            || builtinsName == llvm::StringRef{pylir::Py::Builtins::List.name})
        {
            return getPySequenceType();
        }
        else if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Type.name})
        {
            return getPyTypeType();
        }
        else if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Function.name})
        {
            return getPyFunctionType();
        }
        else if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Str.name})
        {
            return getPyStringType();
        }
        else
        {
            return getPyObjectType();
        }
    }

    mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttr objectAttr)
    {
        auto typeObject = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getAttr());
        PYLIR_ASSERT(typeObject);
        auto slots = typeObject.initializer().getSlots();
        unsigned count = 0;
        auto result = llvm::find_if(slots.getValue(), [](auto pair) { return pair.first.getValue() == "__slots__"; });
        if (result != slots.getValue().end())
        {
            auto tuple = dereference<pylir::Py::TupleAttr>(result->second);
            count = tuple.getValue().size();
        }
        return llvm::TypeSwitch<pylir::Py::ObjectAttr, mlir::LLVM::LLVMStructType>(objectAttr)
            .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr>([&](auto) { return getPySequenceType(count); })
            .Case([&](pylir::Py::StringAttr) { return getPyStringType(count); })
            .Case([&](pylir::Py::TypeAttr) { return getPyTypeType(count); })
            .Case([&](pylir::Py::FunctionAttr) { return getPyFunctionType(count); })
            .Default([&](auto) { return getPyObjectType(count); });
    }

    mlir::Value createSizeOf(mlir::Location loc, mlir::OpBuilder& builder, mlir::Type type)
    {
        auto null = builder.create<mlir::LLVM::NullOp>(loc, mlir::LLVM::LLVMPointerType::get(type));
        auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        auto gep = builder.create<mlir::LLVM::GEPOp>(loc, null.getType(), null, mlir::ValueRange{one});
        return builder.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gep);
    }

    enum class Runtime
    {
        Memcmp,
        pylir_gc_alloc,
    };

    mlir::Value createRuntimeCall(mlir::Location loc, mlir::OpBuilder& builder, Runtime func, mlir::ValueRange args)
    {
        mlir::Type returnType;
        llvm::SmallVector<mlir::Type> argumentTypes;
        std::string functionName;
        // TODO: Some common abstraction to get C Types? Maybe in CABI?
        switch (func)
        {
            case Runtime::Memcmp:
                returnType = builder.getI32Type();
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(builder.getI8Type()),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), builder.getI64Type()};
                functionName = "memcmp";
                break;
            case Runtime::pylir_gc_alloc:
                returnType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
                argumentTypes = {builder.getI64Type()};
                functionName = "pylir_gc_alloc";
                break;
        }
        auto module = mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp());
        auto llvmFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName);
        if (!llvmFunc)
        {
            mlir::OpBuilder::InsertionGuard guard{builder};
            builder.setInsertionPointToEnd(module.getBody());
            llvmFunc = m_cabi->declareFunc(builder, loc, returnType, functionName, argumentTypes);
        }
        return m_cabi->callFunc(builder, loc, llvmFunc, args);
    }

    void initializeGlobal(mlir::LLVM::GlobalOp global, pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
    {
        builder.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), global.getType());
        auto typeObjectAttr =
            m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getValue()).initializer();
        auto typeObj = builder.create<mlir::LLVM::AddressOfOp>(
            global.getLoc(), mlir::LLVM::LLVMPointerType::get(typeOf(typeObjectAttr)), objectAttr.getType());
        auto bitcast = builder.create<mlir::LLVM::BitcastOp>(
            global.getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObjectType()), typeObj);
        undef =
            builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, bitcast, builder.getI32ArrayAttr({0}));
        llvm::TypeSwitch<pylir::Py::ObjectAttr>(objectAttr)
            .Case<pylir::Py::TupleAttr, pylir::Py::ListAttr, pylir::Py::SetAttr, pylir::Py::StringAttr>(
                [&](auto attr)
                {
                    auto values = attr.getValueAttr();
                    auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI64IntegerAttr(values.size()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 0}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 1}));

                    mlir::Type elementType;
                    using AttrType = std::decay_t<decltype(attr)>;
                    if constexpr (std::is_same_v<AttrType, pylir::Py::StringAttr>)
                    {
                        elementType = builder.getI8Type();
                    }
                    else
                    {
                        elementType = mlir::LLVM::LLVMPointerType::get(getPyObjectType());
                    }

                    auto bufferObject = m_globalBuffers.lookup(values);
                    if (!bufferObject)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        if constexpr (std::is_same_v<AttrType, pylir::Py::StringAttr>)
                        {
                            bufferObject = builder.create<mlir::LLVM::GlobalOp>(
                                global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, values.size()), true,
                                mlir::LLVM::Linkage::Private, "buffer$", values, 0, 0, true);
                            bufferObject.setUnnamedAddrAttr(
                                mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                            m_symbolTable.insert(bufferObject);
                            m_globalBuffers.insert({values, bufferObject});
                        }
                        else
                        {
                            bufferObject = builder.create<mlir::LLVM::GlobalOp>(
                                global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, values.size()), true,
                                mlir::LLVM::Linkage::Private, "buffer$", mlir::Attribute{}, 0, 0, true);
                            bufferObject.setUnnamedAddrAttr(
                                mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                            m_symbolTable.insert(bufferObject);
                            m_globalBuffers.insert({values, bufferObject});
                            builder.setInsertionPointToStart(&bufferObject.getInitializerRegion().emplaceBlock());
                            mlir::Value arrayUndef =
                                builder.create<mlir::LLVM::UndefOp>(global.getLoc(), bufferObject.getType());
                            for (auto element : llvm::enumerate(values))
                            {
                                auto constant = getConstant(global.getLoc(), element.value(), builder);
                                arrayUndef = builder.create<mlir::LLVM::InsertValueOp>(
                                    global.getLoc(), arrayUndef, constant,
                                    builder.getI32ArrayAttr({static_cast<std::int32_t>(element.index())}));
                            }
                            builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), arrayUndef);
                        }
                    }
                    auto bufferAddress = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), bufferObject);
                    auto zero = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), builder.getI32Type(),
                                                                       builder.getI32IntegerAttr(0));
                    auto gep = builder.create<mlir::LLVM::GEPOp>(global.getLoc(),
                                                                 mlir::LLVM::LLVMPointerType::get(elementType),
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
            .Case(
                [&](pylir::Py::TypeAttr)
                {
                    auto instanceType = getInstanceType(global.getName());

                    auto sizeOf = createSizeOf(global.getLoc(), builder, instanceType);
                    auto pointerSize = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI32IntegerAttr(getPointerBitwidth() / 8));
                    auto asCount = builder.create<mlir::LLVM::UDivOp>(global.getLoc(), sizeOf, pointerSize);
                    auto oneI = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), getIndexType(),
                                                                       builder.getI32IntegerAttr(1));
                    auto asOffset = builder.create<mlir::LLVM::SubOp>(global.getLoc(), asCount, oneI);
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, asOffset,
                                                                      builder.getI32ArrayAttr({1}));
                })
            .Case(
                [&](pylir::Py::FunctionAttr function)
                {
                    auto address = builder.create<mlir::LLVM::AddressOfOp>(
                        global.getLoc(), convertType(pylir::Py::getUniversalCCType(&getContext())),
                        function.getValue());
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, address,
                                                                      builder.getI32ArrayAttr({1}));
                });
        auto result = llvm::find_if(typeObjectAttr.getSlots().getValue(),
                                    [](auto pair) { return pair.first.getValue() == "__slots__"; });
        if (result != typeObjectAttr.getSlots().getValue().end())
        {
            for (auto slot : llvm::enumerate(dereference<pylir::Py::TupleAttr>(result->second).getValue()))
            {
                auto element = llvm::find_if(
                    objectAttr.getSlots().getValue(), [&](auto pair)
                    { return pair.first.getValue() == slot.value().cast<pylir::Py::StringAttr>().getValue(); });
                mlir::Value value;
                if (element == objectAttr.getSlots().getValue().end())
                {
                    value = builder.create<mlir::LLVM::NullOp>(global.getLoc(),
                                                               mlir::LLVM::LLVMPointerType::get(getPyObjectType()));
                }
                else
                {
                    value = getConstant(global.getLoc(), element->second, builder);
                }
                auto indices = builder.getI32ArrayAttr(
                    {static_cast<int>(global.getType().cast<mlir::LLVM::LLVMStructType>().getBody().size() - 1),
                     static_cast<int>(slot.index())});
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, value, indices);
            }
        }

        builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), undef);
    }

    mlir::Value getConstant(mlir::Location loc, mlir::Attribute attribute, mlir::OpBuilder& builder)
    {
        mlir::LLVM::AddressOfOp address;
        if (auto ref = attribute.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            address = builder.create<mlir::LLVM::AddressOfOp>(
                loc,
                mlir::LLVM::LLVMPointerType::get(
                    typeOf(m_symbolTable.lookup<pylir::Py::GlobalValueOp>(ref.getValue()).initializer())),
                ref);
        }
        else
        {
            address = builder.create<mlir::LLVM::AddressOfOp>(
                loc, createConstant(attribute.cast<pylir::Py::ObjectAttr>(), builder));
        }
        return builder.create<mlir::LLVM::BitcastOp>(loc, mlir::LLVM::LLVMPointerType::get(getPyObjectType()), address);
    }

    mlir::LLVM::GlobalOp createConstant(pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
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

    template <class T>
    T dereference(mlir::Attribute attr)
    {
        if (auto ref = attr.dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            return m_symbolTable.lookup<pylir::Py::GlobalValueOp>(ref.getAttr()).initializer().dyn_cast<T>();
        }
        return attr.dyn_cast<T>();
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
        auto value = getTypeConverter()->getConstant(op.getLoc(), op.constant(), rewriter);
        rewriter.replaceOp(op, {value});
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
        static llvm::DenseSet<llvm::StringRef> immutable = {
            pylir::Py::Builtins::Tuple.name,
            pylir::Py::Builtins::Int.name,
            pylir::Py::Builtins::Float.name,
            pylir::Py::Builtins::Str.name,
        };
        bool constant = op.constant() || immutable.contains(op.initializer().getType().getValue());
        auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(op, type, constant, linkage, op.getName(),
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

struct TupleGetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleGetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleGetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::TupleGetItemOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::TupleGetItemOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.tuple());
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

struct TupleLenOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleLenOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleLenOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::TupleLenOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::TupleLenOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.tuple());
        auto sizePtr = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                          tuple, mlir::ValueRange{zero, one, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, sizePtr);
    }
};

struct FunctionGetFunctionOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::FunctionGetFunctionOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::FunctionGetFunctionOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::FunctionGetFunctionOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::FunctionGetFunctionOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto function = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyFunctionType()), adaptor.function());
        auto funcPtrPtr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getType())), function,
            mlir::ValueRange{zero, one});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, funcPtrPtr);
    }
};

struct GetSlotOpConstantConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GetSlotOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto constant = op.typeObject().getDefiningOp<pylir::Py::ConstantOp>();
        if (!constant)
        {
            return mlir::failure();
        }
        pylir::Py::ObjectAttr typeObject;
        auto ref = constant.constant().dyn_cast<mlir::FlatSymbolRefAttr>();
        typeObject = getTypeConverter()->dereference<pylir::Py::ObjectAttr>(constant.constant());
        if (!typeObject)
        {
            return mlir::failure();
        }

        auto iter = llvm::find_if(typeObject.getSlots().getValue(),
                                  [](auto pair) { return pair.first.getValue() == "__slots__"; });
        if (iter == typeObject.getSlots().getValue().end())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
            return mlir::success();
        }
        auto tupleAttr = getTypeConverter()->dereference<pylir::Py::TupleAttr>(iter->second);
        PYLIR_ASSERT(tupleAttr);
        auto result = llvm::find_if(tupleAttr.getValue(),
                                    [&](mlir::Attribute attribute)
                                    {
                                        auto str = getTypeConverter()->dereference<pylir::Py::StringAttr>(attribute);
                                        PYLIR_ASSERT(str);
                                        return str.getValueAttr() == adaptor.slot();
                                    });
        if (result == tupleAttr.getValue().end())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
            return mlir::success();
        }

        // I could create GEP here to read the offset component of the type object, but LLVM is not aware that the size
        // component is const, even if the rest of the type isn't. So instead we calculate the size here again to have
        // it be a constant. This allows LLVM to lower the whole access to a single GEP + load, which is equal to member
        // access in eg. C or C++
        mlir::Type instanceType;
        if (ref)
        {
            instanceType = getTypeConverter()->getInstanceType(ref.getValue());
        }
        else
        {
            instanceType = getTypeConverter()->getPyObjectType();
        }
        auto sizeOf = getTypeConverter()->createSizeOf(op.getLoc(), rewriter, instanceType);
        mlir::Value i8Ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), adaptor.object());
        i8Ptr = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), i8Ptr.getType(), i8Ptr, sizeOf);
        auto objectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            i8Ptr);
        auto index = createIndexConstant(rewriter, op.getLoc(), result - tupleAttr.getValue().begin());
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), objectPtrPtr.getType(), objectPtrPtr, index);
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, gep);
        return mlir::success();
    }
};

struct GetSlotOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::GetSlotOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::GetSlotOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto block = op->getBlock();
        auto endBlock = op->getBlock()->splitBlock(op);
        endBlock->addArgument(typeConverter->convertType(op.getType()));

        rewriter.setInsertionPointToEnd(block);
        auto str = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), pylir::Py::StringAttr::get(getContext(), adaptor.slot().getValue()));
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple =
            rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.typeObject(), typeRef, "__slots__");
        auto len = rewriter.create<pylir::Py::TupleLenOp>(op.getLoc(), rewriter.getIndexType(), slotsTuple);
        auto condition = new mlir::Block;
        {
            auto zero = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                                                 rewriter.getIndexAttr(0));
            condition->addArgument(getIndexType());
            rewriter.create<mlir::BranchOp>(op.getLoc(), condition, mlir::ValueRange{zero});
        }

        condition->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(condition);
        auto isLess = rewriter.create<mlir::arith::CmpIOp>(op.getLoc(), mlir::arith::CmpIPredicate::ult,
                                                           condition->getArgument(0), len);
        auto unbound = rewriter.create<pylir::Py::ConstantOp>(op.getLoc(), pylir::Py::UnboundAttr::get(getContext()));
        auto body = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(op.getLoc(), isLess, body, endBlock, mlir::ValueRange{unbound});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto element = rewriter.create<pylir::Py::TupleGetItemOp>(op.getLoc(), slotsTuple, condition->getArgument(0));
        auto isEqual = rewriter.create<pylir::Py::StrEqualOp>(op.getLoc(), element, str);
        auto foundIndex = new mlir::Block;
        auto loop = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(op.getLoc(), isEqual, foundIndex, loop, mlir::ValueRange{});
        loop->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(loop);
        {
            auto one = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                                                rewriter.getIndexAttr(1));
            auto increment = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), condition->getArgument(0), one);
            rewriter.create<mlir::BranchOp>(op.getLoc(), condition, mlir::ValueRange{increment});
        }

        foundIndex->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(foundIndex);
        auto typeObj = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeType()), adaptor.typeObject());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      typeObj, mlir::ValueRange{zero, one});
        auto offset = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto index = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, condition->getArgument(0));
        gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            adaptor.object(), mlir::ValueRange{zero, one, index});
        auto slot = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        rewriter.create<mlir::BranchOp>(op.getLoc(), endBlock, mlir::ValueRange{slot});

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, endBlock->getArgument(0));
    }
};

struct StrEqualOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StrEqualOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StrEqualOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::StrEqualOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::StrEqualOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto block = op->getBlock();
        auto endBlock = op->getBlock()->splitBlock(op);
        endBlock->addArgument(rewriter.getI1Type());
        rewriter.setInsertionPointToEnd(block);

        auto sameObject = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, adaptor.lhs(),
                                                              adaptor.rhs());
        auto isNot = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sameObject, endBlock, mlir::ValueRange{sameObject}, isNot,
                                              mlir::ValueRange{});

        isNot->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(isNot);
        auto lhs = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyStringType()), adaptor.lhs());
        auto rhs = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), lhs.getType(), adaptor.rhs());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto lhsGep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                         lhs, mlir::ValueRange{zero, one, zero});
        auto lhsLen = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), lhsGep);
        auto rhsGep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                         rhs, mlir::ValueRange{zero, one, zero});
        auto rhsLen = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rhsGep);
        auto sizeEqual =
            rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, rhsLen);
        auto bufferCmp = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeEqual, bufferCmp, endBlock, mlir::ValueRange{sizeEqual});

        bufferCmp->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(bufferCmp);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        lhsGep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type())), lhs,
            mlir::ValueRange{zero, one, two});
        auto lhsBuffer = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), lhsGep);
        rhsGep =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), lhsGep.getType(), rhs, mlir::ValueRange{zero, one, two});
        auto rhsBuffer = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rhsGep);
        auto result = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::Memcmp,
                                                            {lhsBuffer, rhsBuffer, lhsLen});
        auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, result, zero);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{isZero}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, {endBlock->getArgument(0)});
    }
};

struct GCAllocObjectConstTypeConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto constant = op.typeObj().getDefiningOp<pylir::Py::ConstantOp>();
        if (!constant)
        {
            return mlir::failure();
        }
        pylir::Py::ObjectAttr typeAttr;
        auto ref = constant.constant().dyn_cast<mlir::FlatSymbolRefAttr>();
        typeAttr = getTypeConverter()->dereference<pylir::Py::ObjectAttr>(constant.constant());
        if (!typeAttr)
        {
            return mlir::failure();
        }

        std::size_t slotLen = 0;
        auto iter = llvm::find_if(typeAttr.getSlots().getValue(),
                                  [](auto pair) { return pair.first.getValue() == "__slots__"; });
        if (iter != typeAttr.getSlots().getValue().end())
        {
            slotLen = getTypeConverter()->dereference<pylir::Py::TupleAttr>(iter->second).getValue().size();
        }
        // I could create GEP here to read the offset component of the type object, but LLVM is not aware that the size
        // component is const, even if the rest of the type isn't. So instead we calculate the size here again to have
        // it be a constant.
        mlir::Type instanceType;
        if (ref)
        {
            instanceType = getTypeConverter()->getInstanceType(ref.getValue());
        }
        else
        {
            instanceType = getTypeConverter()->getPyObjectType();
        }
        auto sizeOf = getTypeConverter()->createSizeOf(op.getLoc(), rewriter, instanceType);
        slotLen *= getTypeConverter()->getPointerBitwidth() / 8;
        auto slotSize = createIndexConstant(rewriter, op.getLoc(), slotLen);
        auto inBytes = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), sizeOf, slotSize);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                            PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto object =
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), memory);
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            object, mlir::ValueRange{zero, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.typeObj(), gep);
        return mlir::success();
    }
};

struct GCAllocObjectOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Mem::GCAllocObjectOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto block = op->getBlock();
        auto endBlock = op->getBlock()->splitBlock(op);
        endBlock->addArgument(getIndexType());

        rewriter.setInsertionPointToEnd(block);
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple = rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.typeObj(), typeRef, "__slots__");
        auto hasSlotsBlock = new mlir::Block;
        {
            auto zero = createIndexConstant(rewriter, op.getLoc(), 0);
            auto hasSlots = rewriter.create<pylir::Py::IsUnboundValueOp>(op.getLoc(), slotsTuple);
            rewriter.create<mlir::CondBranchOp>(op.getLoc(), hasSlots, hasSlotsBlock, endBlock, zero);
        }

        hasSlotsBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(hasSlotsBlock);
        auto len = rewriter.create<pylir::Py::TupleLenOp>(op.getLoc(), getIndexType(), slotsTuple);
        rewriter.create<mlir::BranchOp>(op.getLoc(), endBlock, mlir::ValueRange{len});

        rewriter.setInsertionPointToStart(endBlock);
        auto typeObj = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyTypeType()), adaptor.typeObj());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      typeObj, mlir::ValueRange{zero, one});
        auto offset = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, endBlock->getArgument(0));
        auto pointerSize = createIndexConstant(rewriter, op.getLoc(), getTypeConverter()->getPointerBitwidth() / 8);
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, pointerSize);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                            PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto object =
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), memory);
        gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            object, mlir::ValueRange{zero, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.typeObj(), gep);
    }
};

struct InitObjectOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitObjectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
    }
};

struct InitListOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitListOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitListOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Mem::InitListOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Mem::InitListOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto list = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.memory());
        auto size = createIndexConstant(rewriter, op.getLoc(), adaptor.initializer().size());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      list, mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), list,
                                                 mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto sizeOf = getTypeConverter()->createSizeOf(
            op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                            PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto array = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getType())), memory);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), list,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);
        rewriter.replaceOp(op, adaptor.memory());
        for (auto iter : llvm::enumerate(adaptor.initializer()))
        {
            auto offset = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                                  rewriter.getI32IntegerAttr(iter.index()));
            gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), array.getType(), array, mlir::ValueRange{offset});
            rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), iter.value(), gep);
        }
    }
};

struct InitTupleFromListOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTupleFromListOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTupleFromListOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Mem::InitTupleFromListOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Mem::InitTupleFromListOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.memory());
        auto list = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()),
            adaptor.initializer());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      list, mlir::ValueRange{zero, one, zero});
        auto size = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto sizeOf = getTypeConverter()->createSizeOf(
            op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                            PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto array = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getType())), memory);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), tuple,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);
        rewriter.replaceOp(op, adaptor.memory());

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), list,
                                                 mlir::ValueRange{zero, one, two});
        auto listArray = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), array);
        auto listArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), listArray);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, listArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
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

    PylirTypeConverter converter(&getContext(), llvm::Triple(targetTriple.getValue()),
                                 llvm::DataLayout(dataLayout.getValue()), module);
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
    patternSet.insert<TupleGetItemOpConversion>(converter);
    patternSet.insert<TupleLenOpConversion>(converter);
    patternSet.insert<FunctionGetFunctionOpConversion>(converter);
    patternSet.insert<GetSlotOpConstantConversion>(converter, 2);
    patternSet.insert<GetSlotOpConversion>(converter);
    patternSet.insert<StrEqualOpConversion>(converter);
    patternSet.insert<GCAllocObjectOpConversion>(converter);
    patternSet.insert<GCAllocObjectConstTypeConversion>(converter, 2);
    patternSet.insert<InitObjectOpConversion>(converter);
    patternSet.insert<InitListOpConversion>(converter);
    patternSet.insert<InitTupleFromListOpConversion>(converter);
    if (mlir::failed(mlir::applyFullConversion(module, conversionTarget, std::move(patternSet))))
    {
        signalPassFailure();
        return;
    }
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(&getContext(), dataLayout.getValue()));
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(&getContext(), targetTriple.getValue()));
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Mem::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
