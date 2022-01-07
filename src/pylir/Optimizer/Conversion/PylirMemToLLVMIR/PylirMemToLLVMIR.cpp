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
#include <llvm/ADT/StringSet.h>
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

bool needToBeRuntimeInit(pylir::Py::ObjectAttr attr)
{
    // Integer attrs currently need to be runtime init due to memory allocation in libtommath
    // Dict attr need to be runtime init due to the hash calculation
    return attr.isa<pylir::Py::IntAttr, pylir::Py::DictAttr>();
}

class PylirTypeConverter : public mlir::LLVMTypeConverter
{
    llvm::DenseMap<pylir::Py::ObjectAttr, mlir::LLVM::GlobalOp> m_globalConstants;
    llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> m_globalBuffers;
    mlir::SymbolTable m_symbolTable;
    std::unique_ptr<pylir::CABI> m_cabi;
    mlir::LLVM::LLVMFuncOp m_globalInit;
    enum class ExceptionModel
    {
        SEH,
        Dwarf
    };
    ExceptionModel m_exceptionModel;

    mlir::LLVM::LLVMArrayType getSlotEpilogue(unsigned slotSize = 0)
    {
        return mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType()), slotSize);
    }

    mlir::LLVM::LLVMStructType getBufferComponent(mlir::Type elementType)
    {
        return mlir::LLVM::LLVMStructType::getLiteral(
            &getContext(), {getIndexType(), getIndexType(), mlir::LLVM::LLVMPointerType::get(elementType)});
    }

    void appendToGlobalInit(mlir::OpBuilder& builder, llvm::function_ref<void()> section)
    {
        mlir::OpBuilder::InsertionGuard guard{builder};
        if (!m_globalInit)
        {
            builder.setInsertionPointToEnd(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
            m_globalInit = builder.create<mlir::LLVM::LLVMFuncOp>(
                builder.getUnknownLoc(), "$__GLOBAL_INIT__",
                mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(), {}),
                mlir::LLVM::Linkage::Internal, true);
            m_globalInit.addEntryBlock();
        }
        builder.setInsertionPointToEnd(&m_globalInit.back());
        section();
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
                    m_exceptionModel = ExceptionModel::SEH;
                }
                else
                {
                    m_cabi = std::make_unique<pylir::X86_64>(mlir::DataLayout{moduleOp});
                    m_exceptionModel = ExceptionModel::Dwarf;
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

    mlir::LLVM::LLVMStructType getPairType()
    {
        auto pair = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "Pair");
        if (!pair.isInitialized())
        {
            [[maybe_unused]] auto result = pair.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                                         mlir::LLVM::LLVMPointerType::get(getPyObjectType())},
                                                        false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pair;
    }

    mlir::LLVM::LLVMStructType getBucketType()
    {
        auto bucket = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "Bucket");
        if (!bucket.isInitialized())
        {
            [[maybe_unused]] auto result = bucket.setBody({getIndexType(), getIndexType()}, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return bucket;
    }

    mlir::LLVM::LLVMStructType getPyDictType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getBufferComponent(getPairType()), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(getBucketType()), getSlotEpilogue(*slotSize)});
        }
        auto pyDict = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyDict");
        if (!pyDict.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyDict.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getBufferComponent(getPairType()),
                                getIndexType(), mlir::LLVM::LLVMPointerType::get(getBucketType())},
                               false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyDict;
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

    mlir::LLVM::LLVMStructType getMPInt()
    {
        auto mpInt = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "mp_int");
        if (!mpInt.isInitialized())
        {
            [[maybe_unused]] auto result =
                mpInt.setBody({mlir::IntegerType::get(&getContext(), 8 * sizeof(int)),
                               mlir::IntegerType::get(&getContext(), 8 * sizeof(int)),
                               mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(&getContext(), 8)),
                               mlir::IntegerType::get(&getContext(), 8 * sizeof(mp_sign))},
                              false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return mpInt;
    }

    mlir::LLVM::LLVMStructType getPyIntType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getMPInt(), getSlotEpilogue(*slotSize)});
        }
        auto pyType = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyInt");
        if (!pyType.isInitialized())
        {
            [[maybe_unused]] auto result = pyType.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType()), getMPInt(), getSlotEpilogue()}, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyType;
    }

    mlir::LLVM::LLVMStructType getUnwindHeader()
    {
        auto unwindHeader = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "_Unwind_Exception");
        if (!unwindHeader.isInitialized())
        {
            llvm::SmallVector<mlir::Type> header = {mlir::IntegerType::get(&getContext(), 64),
                                                    mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(
                                                        mlir::LLVM::LLVMVoidType::get(&getContext()), {}))};
            switch (m_exceptionModel)
            {
                case ExceptionModel::Dwarf:
                    header.append(
                        {mlir::IntegerType::get(&getContext(), 64), mlir::IntegerType::get(&getContext(), 64)});
                    break;
                case ExceptionModel::SEH:
                    header.push_back(mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(&getContext(), 64), 6));
                    break;
                default: PYLIR_UNREACHABLE;
            }
            [[maybe_unused]] auto result = unwindHeader.setBody(header, false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return unwindHeader;
    }

    mlir::LLVM::LLVMStructType getPyBaseExceptionType(llvm::Optional<unsigned> slotSize = {})
    {
        // While the itanium ABI specifies a 64 bit alignment, GCC and libunwind implementations specify the header
        // to the max alignment (16 bytes on x64). We put the landing pad which is an integer of pointer width
        // right after the PyObjectType base so that the unwind header is always 16 bytes aligned.
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                mlir::IntegerType::get(&getContext(), getPointerBitwidth()), getUnwindHeader(),
                                mlir::IntegerType::get(&getContext(), 32), getSlotEpilogue(*slotSize)});
        }
        auto pyBaseException = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyBaseException");
        if (!pyBaseException.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyBaseException.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                         mlir::IntegerType::get(&getContext(), getPointerBitwidth()), getUnwindHeader(),
                                         mlir::IntegerType::get(&getContext(), 32), getSlotEpilogue()},
                                        false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyBaseException;
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
        else if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Int.name}
                 || builtinsName == llvm::StringRef{pylir::Py::Builtins::Bool.name})
        {
            return getPyIntType();
        }
        else if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Dict.name})
        {
            return getPyDictType();
        }
        else
        {
            static auto builtinExceptionClasses = []
            {
                llvm::StringSet<> set;
#define BUILTIN_EXCEPTION(x, id, ...) set.insert(id);
#define BUILTIN(...)
#include <pylir/Interfaces/Builtins.def>
                return set;
            }();
            if (builtinExceptionClasses.contains(builtinsName))
            {
                return getPyBaseExceptionType();
            }
            return getPyObjectType();
        }
    }

    mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttr objectAttr)
    {
        auto typeObject = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getAttr());
        PYLIR_ASSERT(typeObject);
        PYLIR_ASSERT(!typeObject.isDeclaration() && "Type objects can't be declarations");
        auto slots = typeObject.initializer()->getSlots();
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
            .Case([&](pylir::Py::IntAttr) { return getPyIntType(count); })
            .Case([&](pylir::Py::DictAttr) { return getPyDictType(count); })
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
        malloc,
        realloc,
        mp_init_u64,
        mp_init,
        mp_unpack,
        mp_radix_size_overestimate,
        mp_to_radix,
        pylir_gc_alloc,
        pylir_str_hash,
        pylir_dict_lookup,
        pylir_dict_insert,
        pylir_dict_erase,
        pylir_print,
        pylir_raise,
    };

    mlir::Value createRuntimeCall(mlir::Location loc, mlir::OpBuilder& builder, Runtime func, mlir::ValueRange args)
    {
        mlir::Type returnType;
        llvm::SmallVector<mlir::Type> argumentTypes;
        std::string functionName;
        switch (func)
        {
            case Runtime::Memcmp:
                returnType = m_cabi->getInt(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(builder.getI8Type()),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), getIndexType()};
                functionName = "memcmp";
                break;
            case Runtime::malloc:
                returnType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
                argumentTypes = {getIndexType()};
                functionName = "malloc";
                break;
            case Runtime::realloc:
                returnType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
                argumentTypes = {returnType, getIndexType()};
                functionName = "realloc";
                break;
            case Runtime::pylir_gc_alloc:
                returnType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
                argumentTypes = {getIndexType()};
                functionName = "pylir_gc_alloc";
                break;
            case Runtime::mp_init_u64:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt()), builder.getI64Type()};
                functionName = "mp_init_u64";
                break;
            case Runtime::pylir_str_hash:
                returnType = getIndexType();
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyStringType())};
                functionName = "pylir_str_hash";
                break;
            case Runtime::pylir_print:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyStringType())};
                functionName = "pylir_print";
                break;
            case Runtime::pylir_raise:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyObjectType())};
                functionName = "pylir_raise";
                break;
            case Runtime::mp_init:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt())};
                functionName = "mp_init";
                break;
            case Runtime::mp_unpack:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt()),
                                 getIndexType(),
                                 m_cabi->getInt(&getContext()),
                                 getIndexType(),
                                 m_cabi->getInt(&getContext()),
                                 getIndexType(),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type())};
                functionName = "mp_unpack";
                break;
            case Runtime::mp_radix_size_overestimate:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt()), m_cabi->getInt(&getContext()),
                                 mlir::LLVM::LLVMPointerType::get(getIndexType())};
                functionName = "mp_radix_size_overestimate";
                break;
            case Runtime::mp_to_radix:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt()),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), getIndexType(),
                                 mlir::LLVM::LLVMPointerType::get(getIndexType()), m_cabi->getInt(&getContext())};
                functionName = "mp_to_radix";
                break;
            case Runtime::pylir_dict_lookup:
                returnType = mlir::LLVM::LLVMPointerType::get(getPyObjectType());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType()), returnType};
                functionName = "pylir_dict_lookup";
                break;
            case Runtime::pylir_dict_erase:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType()),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType())};
                functionName = "pylir_dict_erase";
                break;
            case Runtime::pylir_dict_insert:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType()),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType()),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType())};
                functionName = "pylir_dict_insert";
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

    pylir::CABI& getCABI() const
    {
        return *m_cabi;
    }

    void initializeGlobal(mlir::LLVM::GlobalOp global, pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
    {
        builder.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), global.getType());
        auto globalValueOp = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getValue());
        PYLIR_ASSERT(!globalValueOp.isDeclaration() && "Type objects can't be a declaration");
        auto typeObjectAttr = *globalValueOp.initializer();
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
                [&](pylir::Py::IntAttr integer)
                {
                    // TODO: Use host 'size_t'
                    auto result = m_globalBuffers.lookup(integer);
                    if (!result)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        auto bigInt = integer.getValue();
                        auto size = mp_pack_count(&bigInt.getHandle(), 0, sizeof(std::size_t));
                        llvm::SmallVector<std::size_t> data(size);
                        (void)mp_pack(data.data(), data.size(), nullptr, mp_order::MP_LSB_FIRST, sizeof(std::size_t),
                                      MP_BIG_ENDIAN, 0, &bigInt.getHandle());
                        auto elementType = builder.getIntegerType(sizeof(std::size_t) * 8);
                        result = builder.create<mlir::LLVM::GlobalOp>(
                            global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, size), true,
                            mlir::LLVM::Linkage::Private, "buffer$", mlir::Attribute{}, 0, 0, true);
                        result.setUnnamedAddrAttr(
                            mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                        m_symbolTable.insert(result);
                        m_globalBuffers.insert({integer, result});
                        builder.setInsertionPointToStart(&result.getInitializerRegion().emplaceBlock());
                        mlir::Value arrayUndef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), result.getType());
                        for (auto element : llvm::enumerate(data))
                        {
                            auto constant = builder.create<mlir::LLVM::ConstantOp>(
                                global.getLoc(), elementType, builder.getIntegerAttr(elementType, element.value()));
                            arrayUndef = builder.create<mlir::LLVM::InsertValueOp>(
                                global.getLoc(), arrayUndef, constant,
                                builder.getI32ArrayAttr({static_cast<std::int32_t>(element.index())}));
                        }
                        builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), arrayUndef);
                    }
                    auto numElements = result.getType().cast<mlir::LLVM::LLVMArrayType>().getNumElements();
                    appendToGlobalInit(
                        builder,
                        [&]
                        {
                            mlir::Value mpIntPtr;
                            {
                                auto toInit = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), global);

                                auto zero = builder.create<mlir::LLVM::ConstantOp>(
                                    global.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
                                auto one = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), builder.getI32Type(),
                                                                                  builder.getI32IntegerAttr(1));
                                mpIntPtr = builder.create<mlir::LLVM::GEPOp>(
                                    global.getLoc(), mlir::LLVM::LLVMPointerType::get(getMPInt()), toInit,
                                    mlir::ValueRange{zero, one});
                            }

                            createRuntimeCall(global.getLoc(), builder, Runtime::mp_init, {mpIntPtr});
                            auto count = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), getIndexType(),
                                                                                builder.getIndexAttr(numElements));
                            auto intType = m_cabi->getInt(&getContext());
                            auto order = builder.create<mlir::LLVM::ConstantOp>(
                                global.getLoc(), intType, builder.getIntegerAttr(intType, mp_order::MP_LSB_FIRST));
                            auto size = builder.create<mlir::LLVM::ConstantOp>(
                                global.getLoc(), getIndexType(), builder.getIndexAttr(getIndexTypeBitwidth() / 8));
                            auto endian = builder.create<mlir::LLVM::ConstantOp>(
                                global.getLoc(), intType, builder.getIntegerAttr(intType, mp_endian::MP_BIG_ENDIAN));
                            auto zero = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), getIndexType(),
                                                                               builder.getIndexAttr(0));
                            auto buffer = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), result);
                            auto i8 = builder.create<mlir::LLVM::BitcastOp>(
                                global.getLoc(), mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), buffer);
                            createRuntimeCall(global.getLoc(), builder, Runtime::mp_unpack,
                                              {mpIntPtr, count, order, size, endian, zero, i8});
                        });
                })
            .Case(
                [&](pylir::Py::DictAttr dict)
                {
                    auto zeroI = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), getIndexType(),
                                                                        builder.getIndexAttr(0));
                    auto nullPair = builder.create<mlir::LLVM::NullOp>(global.getLoc(),
                                                                       mlir::LLVM::LLVMPointerType::get(getPairType()));
                    auto nullBuckets = builder.create<mlir::LLVM::NullOp>(
                        global.getLoc(), mlir::LLVM::LLVMPointerType::get(getBucketType()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI,
                                                                      builder.getI32ArrayAttr({1, 0}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI,
                                                                      builder.getI32ArrayAttr({1, 1}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, nullPair,
                                                                      builder.getI32ArrayAttr({1, 2}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI,
                                                                      builder.getI32ArrayAttr({2}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, nullBuckets,
                                                                      builder.getI32ArrayAttr({3}));
                    if (dict.getValue().empty())
                    {
                        return;
                    }
                    appendToGlobalInit(
                        builder,
                        [&]
                        {
                            auto address = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), global);
                            auto dictionary = builder.create<mlir::LLVM::BitcastOp>(
                                global.getLoc(), mlir::LLVM::LLVMPointerType::get(getPyDictType()), address);
                            for (auto& [key, value] : dict.getValue())
                            {
                                auto keyValue = getConstant(global.getLoc(), key, builder);
                                auto valueValue = getConstant(global.getLoc(), value, builder);
                                createRuntimeCall(global.getLoc(), builder, Runtime::pylir_dict_insert,
                                                  {dictionary, keyValue, valueValue});
                            }
                        });
                })
            .Case(
                [&](pylir::Py::TypeAttr)
                {
                    auto instanceType = getInstanceType(global.getName());

                    auto sizeOf = createSizeOf(global.getLoc(), builder, instanceType);
                    auto pointerSize = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI32IntegerAttr(getPointerBitwidth() / 8));
                    auto asCount = builder.create<mlir::LLVM::UDivOp>(global.getLoc(), sizeOf, pointerSize);
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, asCount,
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
            auto globalValueOp = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(ref.getValue());
            if (globalValueOp.initializer())
            {
                address = builder.create<mlir::LLVM::AddressOfOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(typeOf(*globalValueOp.initializer())), ref);
            }
            else
            {
                address = builder.create<mlir::LLVM::AddressOfOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(getPyObjectType(0)), ref);
            }
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
        auto globalOp =
            builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), type, !needToBeRuntimeInit(objectAttr),
                                                 mlir::LLVM::Linkage::Private, "const$", mlir::Attribute{}, 0, 0, true);
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
            auto globalValueOp = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(ref.getAttr());
            PYLIR_ASSERT(!globalValueOp.isDeclaration() && "Type objects can't be a declaration");
            return globalValueOp.initializer()->dyn_cast<T>();
        }
        return attr.dyn_cast<T>();
    }

    mlir::LLVM::LLVMFuncOp getGlobalInit()
    {
        return m_globalInit;
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
        mlir::Type type;
        if (op.isDeclaration())
        {
            type = getTypeConverter()->getPyObjectType();
        }
        else
        {
            type = getTypeConverter()->typeOf(*op.initializer());
        }
        mlir::LLVM::Linkage linkage;
        switch (op.getVisibility())
        {
            case mlir::SymbolTable::Visibility::Public: linkage = mlir::LLVM::linkage::Linkage::External; break;
            case mlir::SymbolTable::Visibility::Private: linkage = mlir::LLVM::linkage::Linkage::Internal; break;
            case mlir::SymbolTable::Visibility::Nested: PYLIR_UNREACHABLE;
        }
        if (op.isDeclaration())
        {
            linkage = mlir::LLVM::linkage::Linkage::External;
        }
        static llvm::DenseSet<llvm::StringRef> immutable = {
            pylir::Py::Builtins::Tuple.name,
            pylir::Py::Builtins::Float.name,
            pylir::Py::Builtins::Str.name,
        };
        bool constant = op.constant();
        if (!op.isDeclaration())
        {
            constant = (constant || immutable.contains(op.initializer()->getType().getValue()))
                       && !needToBeRuntimeInit(*op.initializer());
        }
        auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(op, type, constant, linkage, op.getName(),
                                                                        mlir::Attribute{});
        if (!op.isDeclaration())
        {
            getTypeConverter()->initializeGlobal(global, *op.initializer(), rewriter);
        }
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

template <class T>
struct BufferLenOpConversion : public ConvertPylirOpToLLVMPattern<T>
{
    using ConvertPylirOpToLLVMPattern<T>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(T) const override
    {
        return mlir::success();
    }

    void rewrite(T op, typename BufferLenOpConversion<T>::OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPySequenceType()),
            adaptor.input());
        auto sizePtr =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                               tuple, mlir::ValueRange{zero, one, zero});
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, sizePtr);
    }
};

struct ListAppendOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ListAppendOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ListAppendOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::ListAppendOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::ListAppendOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto block = op->getBlock();
        auto endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        rewriter.setInsertionPointToEnd(block);

        auto zeroI32 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto oneI32 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto list = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.list());
        auto sizePtr = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                          list, mlir::ValueRange{zeroI32, oneI32, zeroI32});
        auto size = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), sizePtr);
        auto oneIndex = createIndexConstant(rewriter, op.getLoc(), 1);
        auto incremented = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, oneIndex);
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), incremented, sizePtr);
        auto capacityPtr =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), list,
                                               mlir::ValueRange{zeroI32, oneI32, oneI32});
        auto capacity = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), capacityPtr);
        auto notEnoughCapacity =
            rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ult, capacity, incremented);
        auto growBlock = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), notEnoughCapacity, growBlock, endBlock);

        growBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(growBlock);
        {
            auto twoI32 = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                                  rewriter.getI32IntegerAttr(2));
            mlir::Value newCapacity = rewriter.create<mlir::LLVM::ShlOp>(op.getLoc(), capacity, oneIndex);
            newCapacity = rewriter.create<mlir::LLVM::UMaxOp>(op.getLoc(), newCapacity, incremented);
            auto arrayPtr = rewriter.create<mlir::LLVM::GEPOp>(
                op.getLoc(),
                mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(
                    mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()))),
                list, mlir::ValueRange{zeroI32, oneI32, twoI32});
            auto array = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), arrayPtr);
            auto pyObjectSize = getTypeConverter()->createSizeOf(
                op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()));
            auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), newCapacity, pyObjectSize);
            auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), array);
            auto newMemory = getTypeConverter()->createRuntimeCall(
                op.getLoc(), rewriter, PylirTypeConverter::Runtime::realloc, {arrayI8, inBytes});
            auto newArray = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), array.getType(), newMemory);
            rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), newArray, arrayPtr);
        }
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        auto twoI32 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        auto arrayPtr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(
                mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()))),
            list, mlir::ValueRange{zeroI32, oneI32, twoI32});
        auto array = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), arrayPtr);
        auto offset = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), array.getType(), array, mlir::ValueRange{size});
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.item(), offset);
    }
};

struct DictTryGetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictTryGetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictTryGetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::DictTryGetItemOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::DictTryGetItemOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyDictType()), adaptor.dict());
        auto result = getTypeConverter()->createRuntimeCall(
            op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_lookup, {dict, adaptor.index()});
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), result.getType());
        auto wasFound = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ne, result, null);
        rewriter.replaceOp(op, {result, wasFound});
    }
};

struct DictSetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictSetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictSetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::DictSetItemOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::DictSetItemOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyDictType()), adaptor.dict());
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_insert,
                                              {dict, adaptor.key(), adaptor.value()});
        rewriter.eraseOp(op);
    }
};

struct DictDelItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictDelItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictDelItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::DictDelItemOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::DictDelItemOp op, OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyDictType()), adaptor.dict());
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_erase,
                                              {dict, adaptor.key()});
        rewriter.eraseOp(op);
    }
};

struct BoolToI1OpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::BoolToI1Op>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::BoolToI1Op>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::BoolToI1Op) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::BoolToI1Op op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto boolean = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyIntType()), adaptor.input());
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(/*TODO: int*/ rewriter.getI32Type()), boolean,
            mlir::ValueRange{zero, one, zero});
        auto load = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto zeroI =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), load.getType(), rewriter.getI32IntegerAttr(0));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, load, zeroI);
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

struct ObjectHashOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ObjectHashOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ObjectHashOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ObjectHashOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        // TODO: proper hash
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, typeConverter->convertType(op.getType()),
                                                            adaptor.object());
        return mlir::success();
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
        auto endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
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
        auto sizeEqualBlock = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeEqual, sizeEqualBlock, endBlock,
                                              mlir::ValueRange{sizeEqual});

        sizeEqualBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(sizeEqualBlock);
        auto zeroI = createIndexConstant(rewriter, op.getLoc(), 0);
        auto sizeZero = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, zeroI);
        auto bufferCmp = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeZero, endBlock, mlir::ValueRange{sizeZero}, bufferCmp,
                                              mlir::ValueRange{});

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
        zeroI = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getTypeConverter()->getCABI().getInt(getContext()),
                                                        rewriter.getI32IntegerAttr(0));
        auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, result, zeroI);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{isZero}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, {endBlock->getArgument(0)});
    }
};

struct StrHashOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StrHashOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StrHashOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::StrHashOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::StrHashOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto str = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyStringType()), adaptor.object());
        auto hash = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                          PylirTypeConverter::Runtime::pylir_str_hash, {str});
        rewriter.replaceOp(op, hash);
    }
};

struct PrintOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::PrintOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::PrintOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::PrintOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::PrintOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto str = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyStringType()), adaptor.string());
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_print, {str});
        rewriter.eraseOp(op);
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

struct SetSlotOpConstantConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::SetSlotOp op, OpAdaptor adaptor,
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
            rewriter.eraseOp(op);
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
            rewriter.eraseOp(op);
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), gep);
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
        auto endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArgument(typeConverter->convertType(op.getType()));

        rewriter.setInsertionPointToEnd(block);
        auto str = rewriter.create<pylir::Py::ConstantOp>(op.getLoc(),
                                                          pylir::Py::StringAttr::get(getContext(), adaptor.slot()));
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
        auto pyObjectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            adaptor.object());
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), pyObjectPtrPtr.getType(), pyObjectPtrPtr,
                                                 mlir::ValueRange{index});
        auto slot = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        rewriter.create<mlir::BranchOp>(op.getLoc(), endBlock, mlir::ValueRange{slot});

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, endBlock->getArgument(0));
    }
};

struct SetSlotOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(pylir::Py::SetSlotOp) const override
    {
        return mlir::success();
    }

    void rewrite(pylir::Py::SetSlotOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto block = op->getBlock();
        auto endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});

        rewriter.setInsertionPointToEnd(block);
        auto str = rewriter.create<pylir::Py::ConstantOp>(op.getLoc(),
                                                          pylir::Py::StringAttr::get(getContext(), adaptor.slot()));
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
        auto body = new mlir::Block;
        rewriter.create<mlir::CondBranchOp>(op.getLoc(), isLess, body, endBlock);

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
        auto pyObjectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType())),
            adaptor.object());
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), pyObjectPtrPtr.getType(), pyObjectPtrPtr,
                                                 mlir::ValueRange{index});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.value(), gep);
        rewriter.create<mlir::BranchOp>(op.getLoc(), endBlock);

        rewriter.eraseOp(op);
    }
};

struct RaiseOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::RaiseOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::RaiseOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::RaiseOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_raise,
                                              {adaptor.exception()});
        rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
        return mlir::success();
    }
};

struct InvokeOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::InvokeOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::InvokeOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::InvokeOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        llvm::SmallVector<mlir::Type> resultTypes;
        [[maybe_unused]] auto result = typeConverter->convertTypes(op->getResultTypes(), resultTypes);
        PYLIR_ASSERT(mlir::succeeded(result));
        rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(op, resultTypes, adaptor.calleeAttr(), adaptor.operands(),
                                                          op.happyPath(), adaptor.normalDestOperands(),
                                                          op.exceptionPath(), adaptor.unwindDestOperands());
        return mlir::success();
    }
};

struct InvokeIndirectOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::InvokeIndirectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::InvokeIndirectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::InvokeIndirectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        llvm::SmallVector<mlir::Type> resultTypes;
        [[maybe_unused]] auto result = typeConverter->convertTypes(op->getResultTypes(), resultTypes);
        PYLIR_ASSERT(mlir::succeeded(result));
        llvm::SmallVector<mlir::Value> operands{adaptor.callee()};
        operands.append(adaptor.operands().begin(), adaptor.operands().end());
        rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(op, resultTypes, operands, op.happyPath(),
                                                          adaptor.normalDestOperands(), op.exceptionPath(),
                                                          adaptor.unwindDestOperands());

        return mlir::success();
    }
};

struct LandingPadOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::LandingPadOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::LandingPadOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::LandingPadOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* dest = rewriter.splitBlock(block, mlir::Block::iterator{op});
        rewriter.setInsertionPointToStart(block);

        auto i8Ptr = mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type());
        llvm::SmallVector<mlir::Value> refs;
        {
            mlir::OpBuilder::InsertionGuard guard{rewriter};
            rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
            for (auto iter : adaptor.catchTypes().getAsRange<mlir::FlatSymbolRefAttr>())
            {
                mlir::Value address = getTypeConverter()->getConstant(op.getLoc(), iter, rewriter);
                while (auto cast = address.getDefiningOp<mlir::LLVM::BitcastOp>())
                {
                    address = cast.getArg().getDefiningOp<mlir::LLVM::AddressOfOp>();
                    PYLIR_ASSERT(address);
                }
                refs.emplace_back(rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), i8Ptr, address));
            }
        }
        auto literal = mlir::LLVM::LLVMStructType::getLiteral(getContext(), {i8Ptr, rewriter.getI32Type()});
        auto landingPad = rewriter.create<mlir::LLVM::LandingpadOp>(op.getLoc(), literal, refs);
        mlir::Value exceptionHeader =
            rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), i8Ptr, landingPad, rewriter.getI32ArrayAttr({0}));
        {
            // Itanium ABI mandates a pointer to the exception header be returned by the landing pad.
            // So we need to subtract the offset of the exception header inside of PyBaseException to get to it.
            auto pyBaseException = getTypeConverter()->getPyBaseExceptionType();
            auto unwindHeader = getTypeConverter()->getUnwindHeader();
            static std::size_t index = [&]
            {
                auto body = getTypeConverter()->getPyBaseExceptionType().getBody();
                return std::find(body.begin(), body.end(), unwindHeader) - body.begin();
            }();
            auto null =
                rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(pyBaseException));
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                                rewriter.getI32IntegerAttr(0));
            auto offset = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                                                  rewriter.getI32IntegerAttr(index));
            auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(unwindHeader),
                                                          null, mlir::ValueRange{zero, offset});
            mlir::Value byteOffset = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), getIndexType(), gep);
            auto zeroI = createIndexConstant(rewriter, op.getLoc(), 0);
            byteOffset = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), zeroI, byteOffset);
            exceptionHeader =
                rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), exceptionHeader.getType(), exceptionHeader, byteOffset);
        }
        auto exceptionObject = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()), exceptionHeader);
        auto catchIndex = rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), rewriter.getI32Type(), landingPad,
                                                                      rewriter.getI32ArrayAttr({1}));
        for (auto [type, succ, args] : llvm::zip(refs, op.getSuccessors(), adaptor.branchArgs()))
        {
            auto index = rewriter.create<mlir::LLVM::EhTypeidForOp>(op.getLoc(), rewriter.getI32Type(), type);
            auto isEqual =
                rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, catchIndex, index);
            auto continueSearch = new mlir::Block;
            llvm::SmallVector<mlir::Value> newArgs{exceptionObject};
            newArgs.append(args.begin(), args.end());
            rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isEqual, succ, newArgs, continueSearch,
                                                  mlir::ValueRange{});
            continueSearch->insertBefore(dest);
            rewriter.setInsertionPointToStart(continueSearch);
        }
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{}, dest);
        rewriter.setInsertionPointToStart(dest);
        rewriter.replaceOpWithNewOp<mlir::LLVM::ResumeOp>(op, landingPad);
        return mlir::success();
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
        auto zeroI8 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
        auto falseC =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes, falseC);
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
        auto endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArgument(getIndexType());

        rewriter.setInsertionPointToEnd(block);
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple = rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.typeObj(), typeRef, "__slots__");
        auto hasSlotsBlock = new mlir::Block;
        {
            auto zero = createIndexConstant(rewriter, op.getLoc(), 0);
            auto hasNoSlots = rewriter.create<pylir::Py::IsUnboundValueOp>(op.getLoc(), slotsTuple);
            rewriter.create<mlir::CondBranchOp>(op.getLoc(), hasNoSlots, endBlock, mlir::ValueRange{zero},
                                                hasSlotsBlock, mlir::ValueRange{});
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
        auto zeroI8 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
        auto falseC =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes, falseC);
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

template <class T>
struct InitSequence : public ConvertPylirOpToLLVMPattern<T>
{
    using ConvertPylirOpToLLVMPattern<T>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult match(T) const override
    {
        return mlir::success();
    }

    void rewrite(T op, typename ConvertPylirOpToLLVMPattern<T>::OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto sequence = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPySequenceType()),
            adaptor.memory());
        auto size = this->createIndexConstant(rewriter, op.getLoc(), adaptor.initializer().size());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                               sequence, mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                                 sequence, mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto sizeOf = this->getTypeConverter()->createSizeOf(
            op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyObjectType()));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);
        auto memory = this->getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                                  PylirTypeConverter::Runtime::malloc, {inBytes});
        auto array = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->typeConverter->convertType(op.getType())), memory);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()),
                                                 sequence, mlir::ValueRange{zero, one, two});
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
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::malloc,
                                                            {inBytes});
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

struct InitTuplePopFrontOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePopFrontOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePopFrontOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTuplePopFrontOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.memory());
        auto prevTuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.tuple());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      prevTuple, mlir::ValueRange{zero, one, zero});
        mlir::Value size = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size, oneI);

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto sizeOf = getTypeConverter()->createSizeOf(
            op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::malloc,
                                                            {inBytes});
        auto array = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getType())), memory);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), tuple,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);
        rewriter.replaceOp(op, adaptor.memory());

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()),
                                                 prevTuple, mlir::ValueRange{zero, one, two});
        mlir::Value prevArray = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        prevArray =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), prevArray.getType(), prevArray, mlir::ValueRange{one});
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), array);
        auto prevArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), prevArray);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, prevArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
        return mlir::success();
    }
};

struct InitTuplePrependOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePrependOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePrependOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTuplePrependOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.memory());
        auto prevTuple = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPySequenceType()), adaptor.tuple());

        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                      prevTuple, mlir::ValueRange{zero, one, zero});
        mlir::Value size = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, oneI);

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()), tuple,
                                                 mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto sizeOf = getTypeConverter()->createSizeOf(
            op.getLoc(), rewriter, mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyObjectType()));
        mlir::Value inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);
        auto memory = getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::malloc,
                                                            {inBytes});
        auto array = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(typeConverter->convertType(op.getType())), memory);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), tuple,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.element(), array);
        rewriter.replaceOp(op, adaptor.memory());

        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()),
                                                 prevTuple, mlir::ValueRange{zero, one, two});
        auto prevArray = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        auto arrayPlusOne =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), prevArray.getType(), array, mlir::ValueRange{one});
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), arrayPlusOne);
        auto prevArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()), prevArray);
        inBytes = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), inBytes, sizeOf);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, prevArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
        return mlir::success();
    }
};

struct InitIntOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitIntOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto casted = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyIntType()), adaptor.memory());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto mpIntPointer = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getMPInt()), casted,
            mlir::ValueRange{zero, one});
        auto value = adaptor.initializer();
        if (value.getType() != rewriter.getI64Type())
        {
            value = rewriter.create<mlir::LLVM::ZExtOp>(op.getLoc(), rewriter.getI64Type(), value);
        }
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_init_u64,
                                              {mpIntPointer, value});
        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
    }
};

struct InitStrOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitStrOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto string = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyStringType()),
            adaptor.memory());

        mlir::Value size = this->createIndexConstant(rewriter, op.getLoc(), 0);
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        for (auto iter : adaptor.strings())
        {
            auto iterString = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyStringType()), iter);
            auto gep =
                rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                                   iterString, mlir::ValueRange{zero, one, zero});
            auto sizeLoaded = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
            size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
        }

        auto gep =
            rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                               string, mlir::ValueRange{zero, one, zero});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                                 string, mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, gep);
        auto array = this->getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                                 PylirTypeConverter::Runtime::malloc, {size});
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), string,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);

        size = this->createIndexConstant(rewriter, op.getLoc(), 0);
        for (auto iter : adaptor.strings())
        {
            auto iterString = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyStringType()), iter);
            gep =
                rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getIndexType()),
                                                   iterString, mlir::ValueRange{zero, one, zero});
            auto sizeLoaded = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
            gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()),
                                                     iterString, mlir::ValueRange{zero, one, two});
            auto sourceLoaded = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
            auto dest = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), array.getType(), array, size);
            auto falseC =
                rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
            rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), dest, sourceLoaded, sizeLoaded, falseC);
            size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
        }
        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
    }
};

struct InitStrFromIntOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrFromIntOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrFromIntOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitStrFromIntOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto string = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyStringType()),
            adaptor.memory());
        auto integer = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getPyIntType()), adaptor.integer());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto mpIntPtr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(this->getTypeConverter()->getMPInt()), integer,
            mlir::ValueRange{zero, one});
        auto sizePtr = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(getIndexType()),
                                                          string, mlir::ValueRange{zero, one, zero});
        auto ten = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), getTypeConverter()->getCABI().getInt(getContext()), rewriter.getI32IntegerAttr(10));
        getTypeConverter()->createRuntimeCall(
            op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_radix_size_overestimate, {mpIntPtr, ten, sizePtr});
        auto capacity = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), sizePtr);
        auto array = this->getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter,
                                                                 PylirTypeConverter::Runtime::malloc, {capacity});
        getTypeConverter()->createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_to_radix,
                                              {mpIntPtr, array, capacity, sizePtr, ten});

        // mp_to_radix sadly includes the NULL terminator that it uses in size...
        mlir::Value size = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), sizePtr);
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size, oneI);
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), size, sizePtr);

        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(capacity.getType()),
                                                      string, mlir::ValueRange{zero, one, one});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), capacity, gep);
        auto two =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(2));
        gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(array.getType()), string,
                                                 mlir::ValueRange{zero, one, two});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), array, gep);

        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
    }
};

struct InitFuncOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitFuncOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitFuncOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitFuncOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto casted = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getPyFunctionType()), adaptor.memory());
        auto zero =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto one =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto fpType = typeConverter->convertType(pylir::Py::getUniversalCCType(getContext()));
        auto fp = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(fpType), casted,
                                                     mlir::ValueRange{zero, one});
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), fpType, adaptor.initializer());
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), address, fp);
        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
    }
};

struct InitDictOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitDictOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitDictOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitDictOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.memory());
        return mlir::success();
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
    // For now, map all functions that are private to internal. Public functions are external. In Python code
    // these are all functions that are not __init__
    for (auto iter : module.getOps<mlir::FuncOp>())
    {
        if (iter.isPublic())
        {
            continue;
        }
        iter->setAttr("llvm.linkage",
                      mlir::LLVM::LinkageAttr::get(&getContext(), mlir::LLVM::linkage::Linkage::Internal));
    }

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
    patternSet.insert<BufferLenOpConversion<pylir::Py::TupleLenOp>>(converter);
    patternSet.insert<FunctionGetFunctionOpConversion>(converter);
    patternSet.insert<GetSlotOpConstantConversion>(converter, 2);
    patternSet.insert<GetSlotOpConversion>(converter);
    patternSet.insert<SetSlotOpConstantConversion>(converter, 2);
    patternSet.insert<SetSlotOpConversion>(converter);
    patternSet.insert<StrEqualOpConversion>(converter);
    patternSet.insert<GCAllocObjectOpConversion>(converter);
    patternSet.insert<GCAllocObjectConstTypeConversion>(converter, 2);
    patternSet.insert<InitObjectOpConversion>(converter);
    patternSet.insert<InitSequence<pylir::Mem::InitListOp>>(converter);
    patternSet.insert<InitSequence<pylir::Mem::InitTupleOp>>(converter);
    patternSet.insert<InitTupleFromListOpConversion>(converter);
    patternSet.insert<ListAppendOpConversion>(converter);
    patternSet.insert<BufferLenOpConversion<pylir::Py::ListLenOp>>(converter);
    patternSet.insert<RaiseOpConversion>(converter);
    patternSet.insert<InitIntOpConversion>(converter);
    patternSet.insert<ObjectHashOpConversion>(converter);
    patternSet.insert<StrHashOpConversion>(converter);
    patternSet.insert<InitFuncOpConversion>(converter);
    patternSet.insert<InitDictOpConversion>(converter);
    patternSet.insert<DictTryGetItemOpConversion>(converter);
    patternSet.insert<DictSetItemOpConversion>(converter);
    patternSet.insert<DictDelItemOpConversion>(converter);
    patternSet.insert<BufferLenOpConversion<pylir::Py::DictLenOp>>(converter);
    patternSet.insert<InitStrOpConversion>(converter);
    patternSet.insert<PrintOpConversion>(converter);
    patternSet.insert<InitStrFromIntOpConversion>(converter);
    patternSet.insert<InvokeOpConversion>(converter);
    patternSet.insert<InvokeIndirectOpConversion>(converter);
    patternSet.insert<LandingPadOpConversion>(converter);
    patternSet.insert<BoolToI1OpConversion>(converter);
    patternSet.insert<InitTuplePrependOpConversion>(converter);
    patternSet.insert<InitTuplePopFrontOpConversion>(converter);
    if (mlir::failed(mlir::applyFullConversion(module, conversionTarget, std::move(patternSet))))
    {
        signalPassFailure();
        return;
    }
    auto builder = mlir::OpBuilder::atBlockEnd(module.getBody());
    builder.create<mlir::LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "pylir_personality_function",
        mlir::LLVM::LLVMFunctionType::get(builder.getI32Type(),
                                          {builder.getI32Type(), builder.getI64Type(),
                                           mlir::LLVM::LLVMPointerType::get(builder.getI8Type()),
                                           mlir::LLVM::LLVMPointerType::get(builder.getI8Type())}));
    for (auto iter : module.getOps<mlir::LLVM::LLVMFuncOp>())
    {
        iter.setPersonalityAttr(mlir::FlatSymbolRefAttr::get(&getContext(), "pylir_personality_function"));
    }
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(&getContext(), dataLayout.getValue()));
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(&getContext(), targetTriple.getValue()));
    if (auto globalInit = converter.getGlobalInit())
    {
        builder.setInsertionPointToEnd(&globalInit.back());
        builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});

        builder.setInsertionPointToEnd(module.getBody());
        builder.create<mlir::LLVM::GlobalCtorsOp>(builder.getUnknownLoc(),
                                                  builder.getArrayAttr({mlir::FlatSymbolRefAttr::get(globalInit)}),
                                                  builder.getI32ArrayAttr({65535}));
    }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::Mem::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}
