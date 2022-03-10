
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

// Keep in sync with PylirGC.cpp
constexpr unsigned REF_ADDRESS_SPACE = 1;

bool needToBeRuntimeInit(pylir::Py::ObjectAttr attr)
{
    // Integer attrs currently need to be runtime init due to memory allocation in libtommath
    // Dict attr need to be runtime init due to the hash calculation
    return attr.isa<pylir::Py::IntAttr, pylir::Py::DictAttr>();
}

mlir::LLVM::LLVMPointerType derivePointer(mlir::Type elementType, mlir::Type basePointerType)
{
    return mlir::LLVM::LLVMPointerType::get(elementType,
                                            basePointerType.cast<mlir::LLVM::LLVMPointerType>().getAddressSpace());
}

template <class Type>
struct Model
{
protected:
    mlir::OpBuilder& m_builder;
    mlir::Value m_pointer;
    Type m_elementType;

public:
    Model(mlir::Location, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : m_builder(builder), m_pointer(pointer), m_elementType(elementType.cast<Type>())
    {
    }

    explicit operator mlir::Value() const
    {
        return m_pointer;
    }
};

template <>
struct Model<mlir::LLVM::LLVMStructType>
{
protected:
    mlir::OpBuilder& m_builder;
    mlir::Value m_pointer;
    mlir::LLVM::LLVMStructType m_elementType;

    template <class ResultModel>
    ResultModel field(mlir::Location loc, std::size_t index)
    {
        auto fieldType = m_elementType.getBody()[index];
        return {loc, m_builder,
                m_builder.create<mlir::LLVM::GEPOp>(loc, derivePointer(fieldType, m_pointer.getType()), m_pointer,
                                                    mlir::ValueRange{},
                                                    llvm::ArrayRef<std::int32_t>{0, static_cast<std::int32_t>(index)}),
                fieldType};
    }

public:
    Model(mlir::Location, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : m_builder(builder), m_pointer(pointer), m_elementType(elementType.cast<mlir::LLVM::LLVMStructType>())
    {
    }

    explicit operator mlir::Value() const
    {
        return m_pointer;
    }
};

template <class ElementModel = void>
struct Pointer : Model<mlir::LLVM::LLVMPointerType>
{
    using Model::Model;

    ElementModel load(mlir::Location loc)
    {
        return {loc, m_builder, m_builder.create<mlir::LLVM::LoadOp>(loc, m_elementType, m_pointer),
                m_elementType.getElementType()};
    }

    mlir::LLVM::StoreOp store(mlir::Location loc, mlir::Value value)
    {
        return m_builder.create<mlir::LLVM::StoreOp>(loc, value, this->m_pointer);
    }

    mlir::LLVM::StoreOp store(mlir::Location loc, const ElementModel& model)
    {
        return m_builder.create<mlir::LLVM::StoreOp>(loc, mlir::Value{model}, this->m_pointer);
    }

    Pointer<ElementModel> offset(mlir::Location loc, mlir::Value index)
    {
        return {loc, m_builder, m_builder.create<mlir::LLVM::GEPOp>(loc, m_pointer.getType(), m_pointer, index),
                m_elementType};
    }

    Pointer<ElementModel> offset(mlir::Location loc, std::int32_t index)
    {
        return {loc, m_builder,
                m_builder.create<mlir::LLVM::GEPOp>(loc, m_pointer.getType(), m_pointer, mlir::ValueRange{}, index),
                m_elementType};
    }
};

template <>
struct Pointer<void> : Model<mlir::Type>
{
    using Model::Model;

    mlir::Value load(mlir::Location loc)
    {
        return m_builder.create<mlir::LLVM::LoadOp>(loc, m_elementType, m_pointer);
    }

    mlir::LLVM::StoreOp store(mlir::Location loc, mlir::Value value)
    {
        return m_builder.create<mlir::LLVM::StoreOp>(loc, value, m_pointer);
    }
};

template <class ElementModel = void>
struct Array : Model<mlir::LLVM::LLVMArrayType>
{
    using Model::Model;

    Pointer<ElementModel> at(mlir::Location loc, mlir::Value index)
    {
        return {loc, m_builder,
                m_builder.create<mlir::LLVM::GEPOp>(
                    loc, derivePointer(m_elementType.getElementType(), m_pointer.getType()), m_pointer, index,
                    llvm::ArrayRef<std::int32_t>{0, mlir::LLVM::GEPOp::kDynamicIndex}),
                m_elementType.getElementType()};
    }

    Pointer<ElementModel> at(mlir::Location loc, std::int32_t index)
    {
        return {
            loc, m_builder,
            m_builder.create<mlir::LLVM::GEPOp>(loc, derivePointer(m_elementType.getElementType(), m_pointer.getType()),
                                                m_pointer, mlir::ValueRange{}, llvm::ArrayRef<std::int32_t>{0, index}),
            m_elementType.getElementType()};
    }
};

struct PyObjectModel : Model<mlir::LLVM::LLVMStructType>
{
    using Model::Model;

    auto typePtr(mlir::Location loc)
    {
        return field<Pointer<PyObjectModel>>(loc, 0);
    }
};

struct PyTupleModel : PyObjectModel
{
    PyTupleModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto sizePtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 1);
    }

    auto trailingPtr(mlir::Location loc)
    {
        return field<Array<PyObjectModel>>(loc, 2);
    }
};

struct PyListModel : PyObjectModel
{
    PyListModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto sizePtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 1);
    }

    auto tuplePtr(mlir::Location loc)
    {
        return field<Pointer<PyTupleModel>>(loc, 2);
    }
};

struct BufferComponentModel : Model<mlir::LLVM::LLVMStructType>
{
    using Model::Model;

    auto sizePtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 0);
    }

    auto capacityPtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 1);
    }

    auto elementPtr(mlir::Location loc)
    {
        return field<Pointer<Pointer<>>>(loc, 2);
    }
};

struct PyDictModel : PyObjectModel
{
    PyDictModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto bufferPtr(mlir::Location loc)
    {
        return field<BufferComponentModel>(loc, 1);
    }
};

struct MPIntModel : Model<mlir::LLVM::LLVMStructType>
{
    using Model::Model;

    auto usedPtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 0);
    }
};

struct PyIntModel : PyObjectModel
{
    PyIntModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto mpIntPtr(mlir::Location loc)
    {
        return field<MPIntModel>(loc, 1);
    }
};

struct PyFunctionModel : PyObjectModel
{
    PyFunctionModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto funcPtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 1);
    }
};

struct PyStringModel : PyObjectModel
{
    PyStringModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto bufferPtr(mlir::Location loc)
    {
        return field<BufferComponentModel>(loc, 1);
    }
};

struct PyTypeModel : PyObjectModel
{
    PyTypeModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType)
        : PyObjectModel(
            loc, builder,
            builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(elementType, pointer.getType()), pointer),
            elementType)
    {
    }

    auto offsetPtr(mlir::Location loc)
    {
        return field<Pointer<>>(loc, 1);
    }

    auto layoutPtr(mlir::Location loc)
    {
        return field<Pointer<PyObjectModel>>(loc, 2);
    }

    auto mroPtr(mlir::Location loc)
    {
        return field<Pointer<PyObjectModel>>(loc, 3);
    }
};

class PylirTypeConverter : public mlir::LLVMTypeConverter
{
    llvm::DenseMap<pylir::Py::ObjectAttr, mlir::LLVM::GlobalOp> m_globalConstants;
    llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> m_globalBuffers;
    mlir::SymbolTable m_symbolTable;
    std::unique_ptr<pylir::PlatformABI> m_cabi;
    mlir::LLVM::LLVMFuncOp m_globalInit;
    enum class ExceptionModel
    {
        SEH,
        Dwarf
    };
    ExceptionModel m_exceptionModel;
    mlir::StringAttr m_rootSection;
    mlir::StringAttr m_collectionSection;
    mlir::StringAttr m_constantSection;
    llvm::DenseMap<mlir::Attribute, mlir::FlatSymbolRefAttr> m_layoutTypeCache;

    mlir::LLVM::LLVMArrayType getSlotEpilogue(unsigned slotSize = 0)
    {
        return mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                              slotSize);
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

    bool isSubtype(pylir::Py::TypeAttr subType, mlir::FlatSymbolRefAttr base)
    {
        auto tuple = dereference<pylir::Py::TupleAttr>(subType.getMRO()).getValue();
        return std::any_of(tuple.begin(), tuple.end(), [base](mlir::Attribute attr) { return attr == base; });
    }

public:
    PylirTypeConverter(mlir::MLIRContext* context, const llvm::Triple& triple, llvm::DataLayout dataLayout,
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
        m_rootSection = mlir::StringAttr::get(context, "py_root");
        m_collectionSection = mlir::StringAttr::get(context, "py_coll");
        m_constantSection = mlir::StringAttr::get(context, "py_const");

        for (const auto& iter :
             {pylir::Py::Builtins::Object, pylir::Py::Builtins::Tuple, pylir::Py::Builtins::List,
              pylir::Py::Builtins::Type, pylir::Py::Builtins::Function, pylir::Py::Builtins::Str,
              pylir::Py::Builtins::Int, pylir::Py::Builtins::Dict, pylir::Py::Builtins::BaseException})
        {
            auto ref = mlir::FlatSymbolRefAttr::get(&getContext(), iter.name);
            m_layoutTypeCache[ref] = ref;
        }
    }

    mlir::LLVM::LLVMStructType getPyObjectType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getSlotEpilogue(*slotSize)});
        }
        auto pyObject = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyObject");
        if (!pyObject.isInitialized())
        {
            [[maybe_unused]] auto result = pyObject.setBody(
                {mlir::LLVM::LLVMPointerType::get(pyObject, REF_ADDRESS_SPACE),
                 mlir::LLVM::LLVMArrayType::get(mlir::LLVM::LLVMPointerType::get(pyObject, REF_ADDRESS_SPACE), 0)},
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
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                convertType(pylir::Py::getUniversalCCType(&getContext())), getSlotEpilogue(*slotSize)});
        }
        auto pyFunction = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyFunction");
        if (!pyFunction.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyFunction.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                    convertType(pylir::Py::getUniversalCCType(&getContext())), getSlotEpilogue()},
                                   false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyFunction;
    }

    mlir::LLVM::LLVMStructType getPyTupleType(llvm::Optional<unsigned> length = {})
    {
        if (length)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                                mlir::LLVM::LLVMArrayType::get(
                                    mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), *length)});
        }
        auto pyTuple = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyTuple");
        if (!pyTuple.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyTuple.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                                 mlir::LLVM::LLVMArrayType::get(
                                     mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), 0)},
                                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyTuple;
    }

    mlir::LLVM::LLVMStructType getPyListType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(getPyTupleType(), REF_ADDRESS_SPACE), getSlotEpilogue(*slotSize)});
        }
        auto pyList = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyList");
        if (!pyList.isInitialized())
        {
            [[maybe_unused]] auto result = pyList.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(getPyTupleType(), REF_ADDRESS_SPACE), getSlotEpilogue()},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyList;
    }

    mlir::LLVM::LLVMStructType getPairType()
    {
        auto pair = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "Pair");
        if (!pair.isInitialized())
        {
            [[maybe_unused]] auto result =
                pair.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                              mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE)},
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
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                getBufferComponent(getPairType()), getIndexType(),
                                mlir::LLVM::LLVMPointerType::get(getBucketType()), getSlotEpilogue(*slotSize)});
        }
        auto pyDict = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyDict");
        if (!pyDict.isInitialized())
        {
            [[maybe_unused]] auto result = pyDict.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                 getBufferComponent(getPairType()), getIndexType(), mlir::LLVM::LLVMPointerType::get(getBucketType())},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyDict;
    }

    mlir::LLVM::LLVMStructType getPyStringType(llvm::Optional<unsigned> slotSize = {})
    {
        if (slotSize)
        {
            return mlir::LLVM::LLVMStructType::getLiteral(
                &getContext(),
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                 getBufferComponent(mlir::IntegerType::get(&getContext(), 8)), getSlotEpilogue(*slotSize)});
        }
        auto pyString = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyString");
        if (!pyString.isInitialized())
        {
            [[maybe_unused]] auto result =
                pyString.setBody({mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
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
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getMPInt(),
                                getSlotEpilogue(*slotSize)});
        }
        auto pyType = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyInt");
        if (!pyType.isInitialized())
        {
            [[maybe_unused]] auto result = pyType.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getMPInt(), getSlotEpilogue()},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyType;
    }

    mlir::LLVM::LLVMStructType getUnwindHeaderType()
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
                &getContext(), {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                mlir::IntegerType::get(&getContext(), getPointerBitwidth()), getUnwindHeaderType(),
                                mlir::IntegerType::get(&getContext(), 32), getSlotEpilogue(*slotSize)});
        }
        auto pyBaseException = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyBaseException");
        if (!pyBaseException.isInitialized())
        {
            [[maybe_unused]] auto result = pyBaseException.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                 mlir::IntegerType::get(&getContext(), getPointerBitwidth()), getUnwindHeaderType(),
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
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getSlotEpilogue(*slotSize)});
        }
        auto pyType = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "PyType");
        if (!pyType.isInitialized())
        {
            [[maybe_unused]] auto result = pyType.setBody(
                {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getIndexType(),
                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE), getSlotEpilogue()},
                false);
            PYLIR_ASSERT(mlir::succeeded(result));
        }
        return pyType;
    }

    mlir::LLVM::LLVMStructType getBuiltinsInstanceType(llvm::StringRef builtinsName)
    {
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Object.name})
        {
            return getPyObjectType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Tuple.name})
        {
            return getPyTupleType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::List.name})
        {
            return getPyListType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Type.name})
        {
            return getPyTypeType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Function.name})
        {
            return getPyFunctionType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Str.name})
        {
            return getPyStringType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Int.name})
        {
            return getPyIntType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::Dict.name})
        {
            return getPyDictType();
        }
        if (builtinsName == llvm::StringRef{pylir::Py::Builtins::BaseException.name})
        {
            return getPyBaseExceptionType();
        }
        PYLIR_UNREACHABLE;
    }

    mlir::FlatSymbolRefAttr getLayoutType(pylir::Py::TypeAttr type)
    {
        if (auto result = m_layoutTypeCache.lookup(type))
        {
            return result;
        }
        mlir::FlatSymbolRefAttr winner;
        auto mro = dereference<pylir::Py::TupleAttr>(type.getMRO()).getValue();
        if (mro.empty())
        {
            // TODO: This is a special case that I should probably disallow at one point instead
            winner = mlir::FlatSymbolRefAttr::get(&getContext(), pylir::Py::Builtins::Object.name);
        }
        else
        {
            for (const auto& iter : mro.drop_front())
            {
                mlir::FlatSymbolRefAttr candidate = getLayoutType(iter.cast<mlir::FlatSymbolRefAttr>());
                if (!winner)
                {
                    winner = candidate;
                    continue;
                }
                if (isSubtype(dereference<pylir::Py::TypeAttr>(candidate), winner))
                {
                    winner = candidate;
                }
            }
        }
        m_layoutTypeCache[type] = winner;
        return winner;
    }

    mlir::FlatSymbolRefAttr getLayoutType(mlir::FlatSymbolRefAttr type)
    {
        if (auto result = m_layoutTypeCache.lookup(type))
        {
            return result;
        }
        auto layout = getLayoutType(dereference<pylir::Py::TypeAttr>(type));
        m_layoutTypeCache[type] = layout;
        return layout;
    }

    mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttr objectAttr)
    {
        auto typeObject = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getAttr());
        PYLIR_ASSERT(typeObject);
        PYLIR_ASSERT(!typeObject.isDeclaration() && "Type objects can't be declarations");
        auto slots = typeObject.getInitializer()->getSlots();
        unsigned count = 0;
        auto result = llvm::find_if(slots.getValue(), [](auto pair) { return pair.first.getValue() == "__slots__"; });
        if (result != slots.getValue().end())
        {
            auto tuple = dereference<pylir::Py::TupleAttr>(result->second);
            count = tuple.getValue().size();
        }
        return llvm::TypeSwitch<pylir::Py::ObjectAttr, mlir::LLVM::LLVMStructType>(objectAttr)
            .Case(
                [&](pylir::Py::TupleAttr attr)
                {
                    PYLIR_ASSERT(count == 0);
                    return getPyTupleType(attr.getValue().size());
                })
            .Case([&](pylir::Py::ListAttr) { return getPyListType(count); })
            .Case([&](pylir::Py::StringAttr) { return getPyStringType(count); })
            .Case([&](pylir::Py::TypeAttr) { return getPyTypeType(count); })
            .Case([&](pylir::Py::FunctionAttr) { return getPyFunctionType(count); })
            .Case([&](pylir::Py::IntAttr) { return getPyIntType(count); })
            .Case([&](pylir::Py::DictAttr) { return getPyDictType(count); })
            .Default([&](auto) { return getPyObjectType(count); });
    }

    enum class Runtime
    {
        Memcmp,
        malloc,
        mp_init_u64,
        mp_init,
        mp_unpack,
        mp_radix_size_overestimate,
        mp_to_radix,
        mp_cmp,
        mp_add,
        pylir_gc_alloc,
        pylir_str_hash,
        pylir_int_get,
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
        llvm::SmallVector<llvm::StringRef> passThroughAttributes;
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
            case Runtime::pylir_gc_alloc:
                // TODO: Can probably change this to return a pointer to a PyObject
                returnType = mlir::LLVM::LLVMPointerType::get(builder.getI8Type(), REF_ADDRESS_SPACE);
                argumentTypes = {getIndexType()};
                functionName = "pylir_gc_alloc";
                break;
            case Runtime::mp_init_u64:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE), builder.getI64Type()};
                functionName = "mp_init_u64";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::pylir_str_hash:
                returnType = getIndexType();
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyStringType(), REF_ADDRESS_SPACE)};
                functionName = "pylir_str_hash";
                passThroughAttributes = {"readonly", "gc-leaf-function", "nounwind"};
                break;
            case Runtime::pylir_print:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyStringType(), REF_ADDRESS_SPACE)};
                functionName = "pylir_print";
                passThroughAttributes = {"gc-leaf-function", "nounwind"};
                break;
            case Runtime::pylir_raise:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE)};
                functionName = "pylir_raise";
                passThroughAttributes = {"noreturn"};
                break;
            case Runtime::mp_init:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE)};
                functionName = "mp_init";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::mp_unpack:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 getIndexType(),
                                 m_cabi->getInt(&getContext()),
                                 getIndexType(),
                                 m_cabi->getInt(&getContext()),
                                 getIndexType(),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type())};
                functionName = "mp_unpack";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::mp_radix_size_overestimate:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 m_cabi->getInt(&getContext()),
                                 mlir::LLVM::LLVMPointerType::get(getIndexType(), REF_ADDRESS_SPACE)};
                functionName = "mp_radix_size_overestimate";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::mp_to_radix:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), getIndexType(),
                                 mlir::LLVM::LLVMPointerType::get(getIndexType(), REF_ADDRESS_SPACE),
                                 m_cabi->getInt(&getContext())};
                functionName = "mp_to_radix";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::mp_cmp:
                returnType = m_cabi->getInt(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE)};
                functionName = "mp_cmp";
                passThroughAttributes = {"inaccessiblememonly", "gc-leaf-function", "nounwind"};
                break;
            case Runtime::mp_add:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE)};
                functionName = "mp_add";
                passThroughAttributes = {"gc-leaf-function", "inaccessiblemem_or_argmemonly", "nounwind"};
                break;
            case Runtime::pylir_int_get:
                returnType = mlir::LLVM::LLVMStructType::getLiteral(
                    &getContext(), {getIndexType(), mlir::IntegerType::get(&getContext(), 1)});
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getMPInt(), REF_ADDRESS_SPACE), getIndexType()};
                functionName = "pylir_int_get";
                passThroughAttributes = {"inaccessiblememonly", "gc-leaf-function", "nounwind"};
                break;
            case Runtime::pylir_dict_lookup:
                returnType = mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE);
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType(), REF_ADDRESS_SPACE), returnType};
                functionName = "pylir_dict_lookup";
                break;
            case Runtime::pylir_dict_erase:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE)};
                functionName = "pylir_dict_erase";
                break;
            case Runtime::pylir_dict_insert:
                returnType = mlir::LLVM::LLVMVoidType::get(&getContext());
                argumentTypes = {mlir::LLVM::LLVMPointerType::get(getPyDictType(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE),
                                 mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE)};
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
            if (!passThroughAttributes.empty())
            {
                llvmFunc.setPassthroughAttr(builder.getStrArrayAttr(passThroughAttributes));
            }
        }
        return m_cabi->callFunc(builder, loc, llvmFunc, args);
    }

    pylir::PlatformABI& getPlatformABI() const
    {
        return *m_cabi;
    }

    void initializeGlobal(mlir::LLVM::GlobalOp global, pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder)
    {
        builder.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), global.getType());
        auto globalValueOp = m_symbolTable.lookup<pylir::Py::GlobalValueOp>(objectAttr.getType().getValue());
        PYLIR_ASSERT(globalValueOp);
        PYLIR_ASSERT(!globalValueOp.isDeclaration() && "Type objects can't be a declaration");
        auto typeObjectAttr = *globalValueOp.getInitializer();
        auto typeObj = builder.create<mlir::LLVM::AddressOfOp>(
            global.getLoc(), mlir::LLVM::LLVMPointerType::get(typeOf(typeObjectAttr), REF_ADDRESS_SPACE),
            objectAttr.getType());
        auto bitcast = builder.create<mlir::LLVM::BitcastOp>(
            global.getLoc(), derivePointer(getPyObjectType(), typeObj.getType()), typeObj);
        undef =
            builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, bitcast, builder.getI32ArrayAttr({0}));
        llvm::TypeSwitch<pylir::Py::ObjectAttr>(objectAttr)
            .Case(
                [&](pylir::Py::StringAttr attr)
                {
                    auto values = attr.getValueAttr();
                    auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI64IntegerAttr(values.size()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 0}));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1, 1}));

                    auto elementType = builder.getI8Type();

                    auto bufferObject = m_globalBuffers.lookup(values);
                    if (!bufferObject)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        bufferObject = builder.create<mlir::LLVM::GlobalOp>(
                            global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, values.size()), true,
                            mlir::LLVM::Linkage::Private, "buffer$", values, 0, 0, true);
                        bufferObject.setUnnamedAddrAttr(
                            mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                        m_symbolTable.insert(bufferObject);
                        m_globalBuffers.insert({values, bufferObject});
                    }
                    auto bufferAddress = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), bufferObject);
                    auto gep = builder.create<mlir::LLVM::GEPOp>(
                        global.getLoc(), mlir::LLVM::LLVMPointerType::get(elementType), bufferAddress,
                        mlir::ValueRange{}, llvm::ArrayRef<std::int32_t>{0, 0});
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, gep,
                                                                      builder.getI32ArrayAttr({1, 2}));
                })
            .Case(
                [&](pylir::Py::TupleAttr attr)
                {
                    auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI64IntegerAttr(attr.getValueAttr().size()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1}));
                    for (const auto& iter : llvm::enumerate(attr.getValue()))
                    {
                        auto constant = getConstant(global.getLoc(), iter.value(), builder);
                        undef = builder.create<mlir::LLVM::InsertValueOp>(
                            global.getLoc(), undef, constant,
                            builder.getI32ArrayAttr({2, static_cast<std::int32_t>(iter.index())}));
                    }
                })
            .Case(
                [&](pylir::Py::ListAttr attr)
                {
                    auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), getIndexType(), builder.getI64IntegerAttr(attr.getValueAttr().size()));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                      builder.getI32ArrayAttr({1}));
                    auto tupleObject = m_globalBuffers.lookup(attr);
                    if (!tupleObject)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        tupleObject = builder.create<mlir::LLVM::GlobalOp>(
                            global.getLoc(), getPyTupleType(attr.getValueAttr().size()), true,
                            mlir::LLVM::Linkage::Private, "tuple$", nullptr, 0, REF_ADDRESS_SPACE, true);
                        initializeGlobal(tupleObject, pylir::Py::TupleAttr::get(attr.getValueAttr()), builder);
                        tupleObject.setUnnamedAddrAttr(
                            mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                        m_symbolTable.insert(tupleObject);
                        m_globalBuffers.insert({attr, tupleObject});
                    }
                    auto address = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), tupleObject);
                    auto casted = builder.create<mlir::LLVM::BitcastOp>(
                        global.getLoc(), derivePointer(getPyTupleType(), address.getType()), address);
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, casted,
                                                                      builder.getI32ArrayAttr({2}));
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
                    auto result = m_globalBuffers.lookup(integer);
                    if (!result)
                    {
                        mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                        builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                        auto bigInt = integer.getValue();
                        auto targetSizeTBytes = getPlatformABI().getSizeT(&getContext()).getIntOrFloatBitWidth() / 8;
                        auto size = mp_pack_count(&bigInt.getHandle(), 0, targetSizeTBytes);
                        llvm::SmallVector<std::size_t> data(size);
                        (void)mp_pack(data.data(), data.size(), nullptr, mp_order::MP_LSB_FIRST, targetSizeTBytes,
                                      MP_BIG_ENDIAN, 0, &bigInt.getHandle());
                        auto elementType = getPlatformABI().getSizeT(&getContext());
                        result = builder.create<mlir::LLVM::GlobalOp>(
                            global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, size), true,
                            mlir::LLVM::Linkage::Private, "buffer$", mlir::Attribute{}, 0, 0, true);
                        result.setUnnamedAddrAttr(
                            mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
                        m_symbolTable.insert(result);
                        m_globalBuffers.insert({integer, result});
                        builder.setInsertionPointToStart(&result.getInitializerRegion().emplaceBlock());
                        mlir::Value arrayUndef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), result.getType());
                        for (const auto& element : llvm::enumerate(data))
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
                                mpIntPtr = builder.create<mlir::LLVM::GEPOp>(
                                    global.getLoc(), derivePointer(getMPInt(), toInit.getType()), toInit,
                                    mlir::ValueRange{}, llvm::ArrayRef<std::int32_t>{0, 1});
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
                                global.getLoc(), derivePointer(getPyDictType(), address.getType()), address);
                            for (const auto& [key, value] : dict.getValue())
                            {
                                auto keyValue = getConstant(global.getLoc(), key, builder);
                                auto valueValue = getConstant(global.getLoc(), value, builder);
                                createRuntimeCall(global.getLoc(), builder, Runtime::pylir_dict_insert,
                                                  {dictionary, keyValue, valueValue});
                            }
                        });
                })
            .Case(
                [&](pylir::Py::TypeAttr attr)
                {
                    auto layoutType = getLayoutType(mlir::FlatSymbolRefAttr::get(global));

                    {
                        auto instanceType = getBuiltinsInstanceType(layoutType.getValue());
                        auto asCount = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), getIndexType(),
                            builder.getI32IntegerAttr(getPlatformABI().getSizeOf(instanceType)
                                                      / (getPointerBitwidth() / 8)));
                        undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, asCount,
                                                                          builder.getI32ArrayAttr({1}));
                    }
                    auto layoutRef = getConstant(global.getLoc(), layoutType, builder);
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, layoutRef,
                                                                      builder.getI32ArrayAttr({2}));
                    auto mroConstant = getConstant(global.getLoc(), attr.getMRO(), builder);
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, mroConstant,
                                                                      builder.getI32ArrayAttr({3}));
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
        const auto& map = typeObjectAttr.getSlots().getValue();
        if (auto result = map.find("__slots__"); result != map.end())
        {
            for (const auto& slot : llvm::enumerate(dereference<pylir::Py::TupleAttr>(result->second).getValue()))
            {
                mlir::Value value;
                const auto& initMap = objectAttr.getSlots().getValue();
                if (auto element = initMap.find(slot.value().cast<pylir::Py::StringAttr>().getValue());
                    element == objectAttr.getSlots().getValue().end())
                {
                    value = builder.create<mlir::LLVM::NullOp>(
                        global.getLoc(), mlir::LLVM::LLVMPointerType::get(getPyObjectType(), REF_ADDRESS_SPACE));
                }
                else
                {
                    value = getConstant(global.getLoc(), element->second, builder);
                }
                auto indices = builder.getI32ArrayAttr(
                    {static_cast<std::int32_t>(global.getType().cast<mlir::LLVM::LLVMStructType>().getBody().size()
                                               - 1),
                     static_cast<std::int32_t>(slot.index())});
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
            if (globalValueOp.getInitializer())
            {
                address = builder.create<mlir::LLVM::AddressOfOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(typeOf(*globalValueOp.getInitializer()), REF_ADDRESS_SPACE),
                    ref);
            }
            else
            {
                address = builder.create<mlir::LLVM::AddressOfOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(getPyObjectType(0), REF_ADDRESS_SPACE), ref);
            }
        }
        else
        {
            address = builder.create<mlir::LLVM::AddressOfOp>(
                loc, createConstant(attribute.cast<pylir::Py::ObjectAttr>(), builder));
        }
        return builder.create<mlir::LLVM::BitcastOp>(loc, derivePointer(getPyObjectType(), address.getType()), address);
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
            builder.getUnknownLoc(), type, !needToBeRuntimeInit(objectAttr), mlir::LLVM::Linkage::Private, "const$",
            mlir::Attribute{}, 0, REF_ADDRESS_SPACE, true);
        globalOp.setUnnamedAddrAttr(mlir::LLVM::UnnamedAddrAttr::get(&getContext(), mlir::LLVM::UnnamedAddr::Global));
        globalOp.setSectionAttr(globalOp.getConstant() ? getConstantSection() : getCollectionSection());
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
            return globalValueOp.getInitializer()->dyn_cast<T>();
        }
        return attr.dyn_cast<T>();
    }

    mlir::LLVM::LLVMFuncOp getGlobalInit()
    {
        return m_globalInit;
    }

    mlir::StringAttr getRootSection() const
    {
        return m_rootSection;
    }

    mlir::StringAttr getCollectionSection() const
    {
        return m_collectionSection;
    }

    mlir::StringAttr getConstantSection() const
    {
        return m_constantSection;
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
    [[nodiscard]] mlir::LLVM::LLVMStructType getPyObjectType() const
    {
        return getTypeConverter()->getPyObjectType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyFunctionType() const
    {
        return getTypeConverter()->getPyFunctionType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyTupleType() const
    {
        return getTypeConverter()->getPyTupleType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyListType() const
    {
        return getTypeConverter()->getPyListType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPairType() const
    {
        return getTypeConverter()->getPairType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getBucketType() const
    {
        return getTypeConverter()->getBucketType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyDictType() const
    {
        return getTypeConverter()->getPyDictType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyStringType() const
    {
        return getTypeConverter()->getPyStringType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getMPIntType() const
    {
        return getTypeConverter()->getMPIntType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyIntType() const
    {
        return getTypeConverter()->getPyIntType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getUnwindHeaderType() const
    {
        return getTypeConverter()->getUnwindHeaderType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyBaseExceptionType() const
    {
        return getTypeConverter()->getPyBaseExceptionType();
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType getPyTypeType() const
    {
        return getTypeConverter()->getPyTypeType();
    }

    [[nodiscard]] mlir::Value getConstant(mlir::Location loc, mlir::Attribute attribute, mlir::OpBuilder& builder) const
    {
        return getTypeConverter()->getConstant(loc, attribute, builder);
    }

    [[nodiscard]] mlir::LLVM::LLVMStructType typeOf(pylir::Py::ObjectAttr objectAttr) const
    {
        return getTypeConverter()->typeOf(objectAttr);
    }

    void initializeGlobal(mlir::LLVM::GlobalOp global, pylir::Py::ObjectAttr objectAttr, mlir::OpBuilder& builder) const
    {
        getTypeConverter()->initializeGlobal(global, objectAttr, builder);
    }

    [[nodiscard]] mlir::LLVM::LLVMPointerType pointer(mlir::Type type, unsigned addrSpace = 0) const
    {
        return mlir::LLVM::LLVMPointerType::get(type, addrSpace);
    }

    [[nodiscard]] PyObjectModel pyObjectModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyObjectType()};
    }

    [[nodiscard]] PyListModel pyListModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyListType()};
    }

    [[nodiscard]] PyTupleModel pyTupleModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyTupleType()};
    }

    [[nodiscard]] PyDictModel pyDictModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyDictType()};
    }

    [[nodiscard]] PyIntModel pyIntModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyIntType()};
    }

    [[nodiscard]] PyFunctionModel pyFunctionModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyFunctionType()};
    }

    [[nodiscard]] PyStringModel pyStringModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyStringType()};
    }

    [[nodiscard]] PyTypeModel pyTypeModel(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value value) const
    {
        return {loc, builder, value, getPyTypeType()};
    }

    [[nodiscard]] std::size_t sizeOf(mlir::Type type) const
    {
        return getTypeConverter()->getPlatformABI().getSizeOf(type);
    }

    [[nodiscard]] std::size_t alignOf(mlir::Type type) const
    {
        return getTypeConverter()->getPlatformABI().getAlignOf(type);
    }

    [[nodiscard]] mlir::Type getInt() const
    {
        return getTypeConverter()->getPlatformABI().getInt(this->getContext());
    }

    mlir::Value createRuntimeCall(mlir::Location loc, mlir::OpBuilder& builder, PylirTypeConverter::Runtime func,
                                  mlir::ValueRange args) const
    {
        return getTypeConverter()->createRuntimeCall(loc, builder, func, args);
    }

    template <class Attr>
    [[nodiscard]] Attr dereference(mlir::Attribute attr) const
    {
        return getTypeConverter()->template dereference<Attr>(attr);
    }

    [[nodiscard]] mlir::Type getBuiltinsInstanceType(mlir::FlatSymbolRefAttr ref) const
    {
        return getTypeConverter()->getBuiltinsInstanceType(ref.getValue());
    }

    [[nodiscard]] mlir::FlatSymbolRefAttr getLayoutType(mlir::FlatSymbolRefAttr ref) const
    {
        return getTypeConverter()->getLayoutType(ref);
    }

    [[nodiscard]] mlir::FlatSymbolRefAttr getLayoutType(pylir::Py::TypeAttr attr) const
    {
        return getTypeConverter()->getLayoutType(attr);
    }

    [[nodiscard]] mlir::StringAttr getRootSection() const
    {
        return getTypeConverter()->getRootSection();
    }

    [[nodiscard]] mlir::StringAttr getCollectionSection() const
    {
        return getTypeConverter()->getCollectionSection();
    }

    [[nodiscard]] mlir::StringAttr getConstantSection() const
    {
        return getTypeConverter()->getConstantSection();
    }

    [[nodiscard]] mlir::Value unrealizedConversion(mlir::OpBuilder& builder, mlir::Value value) const
    {
        return builder
            .create<mlir::UnrealizedConversionCastOp>(value.getLoc(), this->typeConverter->convertType(value.getType()),
                                                      value)
            .getResult(0);
    }

private:
    [[nodiscard]] PylirTypeConverter* getTypeConverter() const
    {
        return static_cast<PylirTypeConverter*>(this->typeConverter);
    }
};

struct ConstantOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ConstantOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ConstantOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ConstantOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (auto unbound = adaptor.getConstant().dyn_cast<pylir::Py::UnboundAttr>())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
            return mlir::success();
        }
        auto value = getConstant(op.getLoc(), op.getConstant(), rewriter);
        rewriter.replaceOp(op, {value});
        return mlir::success();
    }
};

struct GlobalValueOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GlobalValueOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GlobalValueOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GlobalValueOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Type type;
        if (op.isDeclaration())
        {
            type = getPyObjectType();
        }
        else
        {
            type = typeOf(*op.getInitializer());
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
        bool constant = op.getConstant();
        if (!op.isDeclaration())
        {
            constant = (constant || immutable.contains(op.getInitializer()->getType().getValue()))
                       && !needToBeRuntimeInit(*op.getInitializer());
        }
        auto global = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(op, type, constant, linkage, op.getName(),
                                                                        mlir::Attribute{}, 0, REF_ADDRESS_SPACE, true);
        if (!op.isDeclaration())
        {
            if (!global.getConstant())
            {
                global.setSectionAttr(getCollectionSection());
            }
            else
            {
                global.setSectionAttr(getConstantSection());
            }
            initializeGlobal(global, *op.getInitializer(), rewriter);
        }
        return mlir::success();
    }
};

struct GlobalHandleOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GlobalHandleOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GlobalHandleOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GlobalHandleOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::LLVM::Linkage linkage;
        switch (op.getVisibility())
        {
            case mlir::SymbolTable::Visibility::Public: linkage = mlir::LLVM::linkage::Linkage::External; break;
            case mlir::SymbolTable::Visibility::Private: linkage = mlir::LLVM::linkage::Linkage::Private; break;
            case mlir::SymbolTable::Visibility::Nested: PYLIR_UNREACHABLE;
        }
        auto global =
            rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(op, pointer(getPyObjectType(), REF_ADDRESS_SPACE), false,
                                                              linkage, op.getName(), mlir::Attribute{}, 0, 0, true);
        global.setSectionAttr(getRootSection());
        rewriter.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), global.getType());
        rewriter.create<mlir::LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange{null});
        return mlir::success();
    }
};

struct LoadOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::LoadOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::LoadOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::LoadOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
            op.getLoc(), pointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE)), op.getHandleAttr());
        rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, address);
        return mlir::success();
    }
};

struct StoreOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StoreOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StoreOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::StoreOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
            op.getLoc(), pointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE)), adaptor.getHandle());
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(), address);
        return mlir::success();
    }
};

struct IsOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IsOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IsOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IsOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::eq, adaptor.getLhs(),
                                                        adaptor.getRhs());
        return mlir::success();
    }
};

struct IsUnboundValueOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IsUnboundValueOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IsUnboundValueOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IsUnboundValueOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), adaptor.getValue().getType());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::eq, adaptor.getValue(), null);
        return mlir::success();
    }
};

struct TypeOfOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TypeOfOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TypeOfOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TypeOfOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto model = pyObjectModel(op.getLoc(), rewriter, adaptor.getObject());
        rewriter.replaceOp(op, mlir::Value{model.typePtr(op.getLoc()).load(op.getLoc())});
        return mlir::success();
    }
};

struct TupleGetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleGetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleGetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TupleGetItemOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, mlir::Value{pyTupleModel(op.getLoc(), rewriter, adaptor.getTuple())
                                               .trailingPtr(op.getLoc())
                                               .at(op.getLoc(), adaptor.getIndex())
                                               .load(op.getLoc())});
        return mlir::success();
    }
};

struct DictLenOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictLenOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::DictLenOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto sequence = this->pyDictModel(op.getLoc(), rewriter, adaptor.getInput());
        rewriter.replaceOp(op, sequence.bufferPtr(op.getLoc()).sizePtr(op.getLoc()).load(op.getLoc()));
        return mlir::success();
    }
};

struct TupleLenOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TupleLenOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TupleLenOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TupleLenOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(
            op, pyTupleModel(op.getLoc(), rewriter, adaptor.getInput()).sizePtr(op.getLoc()).load(op.getLoc()));
        return mlir::success();
    }
};

struct ListLenOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ListLenOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ListLenOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ListLenOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, pyListModel(op.getLoc(), rewriter, adaptor.getInput())
                                   .tuplePtr(op.getLoc())
                                   .load(op.getLoc())
                                   .sizePtr(op.getLoc())
                                   .load(op.getLoc()));
        return mlir::success();
    }
};

struct ListAppendOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ListAppendOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ListAppendOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ListAppendOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        rewriter.setInsertionPointToEnd(block);

        auto list = pyListModel(op.getLoc(), rewriter, adaptor.getList());
        auto tuplePtr = list.tuplePtr(op.getLoc()).load(op.getLoc());
        auto sizePtr = list.sizePtr(op.getLoc());
        auto size = sizePtr.load(op.getLoc());
        auto oneIndex = createIndexConstant(rewriter, op.getLoc(), 1);
        auto incremented = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, oneIndex);

        auto capacityPtr = tuplePtr.sizePtr(op.getLoc());
        auto capacity = capacityPtr.load(op.getLoc());
        auto notEnoughCapacity =
            rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ult, capacity, incremented);
        auto* growBlock = new mlir::Block;
        auto* notGrowBlock = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), notEnoughCapacity, growBlock, notGrowBlock);

        notGrowBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(notGrowBlock);
        tuplePtr.trailingPtr(op.getLoc()).at(op.getLoc(), size).store(op.getLoc(), adaptor.getItem());
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{}, endBlock);

        growBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(growBlock);
        {
            mlir::Value newCapacity = rewriter.create<mlir::LLVM::ShlOp>(op.getLoc(), capacity, oneIndex);
            newCapacity = rewriter.create<mlir::LLVM::UMaxOp>(op.getLoc(), newCapacity, incremented);

            mlir::Value tupleMemory = rewriter.create<pylir::Mem::GCAllocTupleOp>(
                op.getLoc(), mlir::Value{tuplePtr.typePtr(op.getLoc()).load(op.getLoc())}, newCapacity);
            tupleMemory = unrealizedConversion(rewriter, tupleMemory);

            auto newTupleModel = pyTupleModel(op.getLoc(), rewriter, tupleMemory);
            newTupleModel.sizePtr(op.getLoc()).store(op.getLoc(), newCapacity);
            auto trailingPtr = newTupleModel.trailingPtr(op.getLoc());
            trailingPtr.at(op.getLoc(), size).store(op.getLoc(), adaptor.getItem());
            auto array = mlir::Value{trailingPtr.at(op.getLoc(), 0)};
            auto prevArray = mlir::Value{tuplePtr.trailingPtr(op.getLoc()).at(op.getLoc(), 0)};
            auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), derivePointer(rewriter.getI8Type(), array.getType()), array);
            auto prevArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), derivePointer(rewriter.getI8Type(), prevArray.getType()), prevArray);
            auto elementTypeSize = createIndexConstant(rewriter, op.getLoc(), sizeOf(pointer(getPyObjectType())));
            auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, elementTypeSize);
            rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), arrayI8, prevArrayI8, inBytes,
                                                  rewriter.create<mlir::LLVM::ConstantOp>(
                                                      op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
            list.tuplePtr(op.getLoc()).store(op.getLoc(), mlir::Value{newTupleModel});
        }
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        sizePtr.store(op.getLoc(), incremented);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct DictTryGetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictTryGetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictTryGetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::DictTryGetItemOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = pyDictModel(op.getLoc(), rewriter, adaptor.getDict());
        auto result = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_lookup,
                                        {mlir::Value{dict}, adaptor.getKey()});
        auto null = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), result.getType());
        auto wasFound = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ne, result, null);
        rewriter.replaceOp(op, {result, wasFound});
        return mlir::success();
    }
};

struct DictSetItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictSetItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictSetItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::DictSetItemOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = pyDictModel(op.getLoc(), rewriter, adaptor.getDict());
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_insert,
                          {mlir::Value{dict}, adaptor.getKey(), adaptor.getValue()});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct DictDelItemOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::DictDelItemOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::DictDelItemOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::DictDelItemOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto dict = pyDictModel(op.getLoc(), rewriter, adaptor.getDict());
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_dict_erase,
                          {mlir::Value{dict}, adaptor.getKey()});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct IntGetIntegerOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IntToIntegerOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IntToIntegerOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IntToIntegerOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto pyInt = pyIntModel(op.getLoc(), rewriter, adaptor.getInput());
        auto mpInt = pyInt.mpIntPtr(op.getLoc());
        auto converted = typeConverter->convertType(op.getResult().getType());
        auto size = createIndexConstant(rewriter, op.getLoc(), sizeOf(converted));
        auto result = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_int_get,
                                        {mlir::Value{mpInt}, size});
        mlir::Value first = rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), getIndexType(), result,
                                                                        rewriter.getI32ArrayAttr({0}));
        auto second = rewriter.create<mlir::LLVM::ExtractValueOp>(op.getLoc(), rewriter.getI1Type(), result,
                                                                  rewriter.getI32ArrayAttr({1}));
        if (first.getType() != converted)
        {
            first = rewriter.create<mlir::LLVM::TruncOp>(op.getLoc(), converted, first);
        }
        rewriter.replaceOp(op, {first, second});
        return mlir::success();
    }
};

struct InitIntAddOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntAddOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntAddOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitIntAddOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto memoryInt = pyIntModel(op.getLoc(), rewriter, adaptor.getMemory()).mpIntPtr(op.getLoc());
        auto lhsInt = pyIntModel(op.getLoc(), rewriter, adaptor.getLhs()).mpIntPtr(op.getLoc());
        auto rhsInt = pyIntModel(op.getLoc(), rewriter, adaptor.getRhs()).mpIntPtr(op.getLoc());

        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_init, mlir::Value{memoryInt});
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_add,
                          {mlir::Value{lhsInt}, mlir::Value{rhsInt}, mlir::Value{memoryInt}});

        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct IntCmpOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::IntCmpOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::IntCmpOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::IntCmpOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto lhsInt = pyIntModel(op.getLoc(), rewriter, adaptor.getLhs()).mpIntPtr(op.getLoc());
        auto rhsInt = pyIntModel(op.getLoc(), rewriter, adaptor.getRhs()).mpIntPtr(op.getLoc());
        auto result = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_cmp,
                                        {mlir::Value{lhsInt}, mlir::Value{rhsInt}});
        mp_ord mpOrd;
        mlir::LLVM::ICmpPredicate predicate;
        switch (adaptor.getPred())
        {
            case pylir::Py::IntCmpKind::eq:
                mpOrd = MP_EQ;
                predicate = mlir::LLVM::ICmpPredicate::eq;
                break;
            case pylir::Py::IntCmpKind::ne:
                mpOrd = MP_EQ;
                predicate = mlir::LLVM::ICmpPredicate::ne;
                break;
            case pylir::Py::IntCmpKind::lt:
                mpOrd = MP_LT;
                predicate = mlir::LLVM::ICmpPredicate::eq;
                break;
            case pylir::Py::IntCmpKind::le:
                mpOrd = MP_GT;
                predicate = mlir::LLVM::ICmpPredicate::ne;
                break;
            case pylir::Py::IntCmpKind::gt:
                mpOrd = MP_GT;
                predicate = mlir::LLVM::ICmpPredicate::eq;
                break;
            case pylir::Py::IntCmpKind::ge:
                mpOrd = MP_LT;
                predicate = mlir::LLVM::ICmpPredicate::ne;
                break;
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, predicate, result,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getInt(), mlir::IntegerAttr::get(getInt(), mpOrd)));
        return mlir::success();
    }
};

struct BoolToI1OpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::BoolToI1Op>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::BoolToI1Op>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::BoolToI1Op op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto load = pyIntModel(op.getLoc(), rewriter, adaptor.getInput())
                        .mpIntPtr(op.getLoc())
                        .usedPtr(op.getLoc())
                        .load(op.getLoc());
        auto zeroI =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), load.getType(), rewriter.getI32IntegerAttr(0));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op, mlir::LLVM::ICmpPredicate::ne, load, zeroI);
        return mlir::success();
    }
};

struct FunctionGetFunctionOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::FunctionGetFunctionOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::FunctionGetFunctionOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::FunctionGetFunctionOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(
            op,
            mlir::Value{
                pyFunctionModel(op.getLoc(), rewriter, adaptor.getFunction()).funcPtr(op.getLoc()).load(op.getLoc())});
        return mlir::success();
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
                                                            adaptor.getObject());
        return mlir::success();
    }
};

struct ObjectIdOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::ObjectIdOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::ObjectIdOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::ObjectIdOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, typeConverter->convertType(op.getType()),
                                                            adaptor.getObject());
        return mlir::success();
    }
};

struct TypeMROOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::TypeMROOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::TypeMROOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::TypeMROOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(
            op, mlir::Value{
                    pyTypeModel(op.getLoc(), rewriter, adaptor.getTypeObject()).mroPtr(op.getLoc()).load(op.getLoc())});
        return mlir::success();
    }
};

struct StrEqualOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StrEqualOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StrEqualOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::StrEqualOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArgument(rewriter.getI1Type(), op.getLoc());
        rewriter.setInsertionPointToEnd(block);

        auto sameObject = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                                                              adaptor.getLhs(), adaptor.getRhs());
        auto* isNot = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sameObject, endBlock, mlir::ValueRange{sameObject}, isNot,
                                              mlir::ValueRange{});

        isNot->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(isNot);
        auto lhs = pyStringModel(op.getLoc(), rewriter, adaptor.getLhs()).bufferPtr(op.getLoc());
        auto rhs = pyStringModel(op.getLoc(), rewriter, adaptor.getRhs()).bufferPtr(op.getLoc());
        auto lhsLen = lhs.sizePtr(op.getLoc()).load(op.getLoc());
        auto rhsLen = rhs.sizePtr(op.getLoc()).load(op.getLoc());
        auto sizeEqual =
            rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, rhsLen);
        auto* sizeEqualBlock = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeEqual, sizeEqualBlock, endBlock,
                                              mlir::ValueRange{sizeEqual});

        sizeEqualBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(sizeEqualBlock);
        auto zeroI = createIndexConstant(rewriter, op.getLoc(), 0);
        auto sizeZero = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, lhsLen, zeroI);
        auto* bufferCmp = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), sizeZero, endBlock, mlir::ValueRange{sizeZero}, bufferCmp,
                                              mlir::ValueRange{});

        bufferCmp->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(bufferCmp);
        auto lhsBuffer = mlir::Value{lhs.elementPtr(op.getLoc()).load(op.getLoc())};
        auto rhsBuffer = mlir::Value{rhs.elementPtr(op.getLoc()).load(op.getLoc())};
        auto result = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::Memcmp,
                                        {lhsBuffer, rhsBuffer, lhsLen});
        zeroI = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getInt(), rewriter.getI32IntegerAttr(0));
        auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::eq, result, zeroI);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{isZero}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, {endBlock->getArgument(0)});
        return mlir::success();
    }
};

struct StrHashOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::StrHashOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::StrHashOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::StrHashOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto str = pyStringModel(op.getLoc(), rewriter, adaptor.getObject());
        auto hash =
            createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_str_hash, mlir::Value{str});
        rewriter.replaceOp(op, hash);
        return mlir::success();
    }
};

struct PrintOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::PrintOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::PrintOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::PrintOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto str = pyStringModel(op.getLoc(), rewriter, adaptor.getString());
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_print, mlir::Value{str});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct GetSlotOpConstantConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GetSlotOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto constant = op.getTypeObject().getDefiningOp<pylir::Py::ConstantOp>();
        if (!constant)
        {
            return mlir::failure();
        }
        pylir::Py::TypeAttr typeObject;
        auto ref = constant.getConstant().dyn_cast<mlir::FlatSymbolRefAttr>();
        typeObject = dereference<pylir::Py::TypeAttr>(constant.getConstant());
        if (!typeObject)
        {
            return mlir::failure();
        }

        auto& map = typeObject.getSlots().getValue();
        auto iter = map.find("__slots__");
        if (iter == map.end())
        {
            rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(op, typeConverter->convertType(op.getType()));
            return mlir::success();
        }
        auto tupleAttr = dereference<pylir::Py::TupleAttr>(iter->second);
        PYLIR_ASSERT(tupleAttr);
        auto* result = llvm::find_if(tupleAttr.getValue(),
                                     [&](mlir::Attribute attribute)
                                     {
                                         auto str = dereference<pylir::Py::StringAttr>(attribute);
                                         PYLIR_ASSERT(str);
                                         return str.getValueAttr() == adaptor.getSlot();
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
            instanceType = getBuiltinsInstanceType(getLayoutType(ref));
        }
        else
        {
            instanceType = getBuiltinsInstanceType(getLayoutType(typeObject));
        }
        mlir::Value i8Ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), adaptor.getObject().getType()), adaptor.getObject());
        i8Ptr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), i8Ptr.getType(), i8Ptr, mlir::ValueRange{},
            llvm::ArrayRef<std::int32_t>{static_cast<std::int32_t>(sizeOf(instanceType))});
        auto objectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE), i8Ptr.getType()), i8Ptr);
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), objectPtrPtr.getType(), objectPtrPtr, mlir::ValueRange{},
            llvm::ArrayRef<std::int32_t>{static_cast<std::int32_t>(result - tupleAttr.getValue().begin())});
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
        auto constant = op.getTypeObject().getDefiningOp<pylir::Py::ConstantOp>();
        if (!constant)
        {
            return mlir::failure();
        }
        pylir::Py::TypeAttr typeObject;
        auto ref = constant.getConstant().dyn_cast<mlir::FlatSymbolRefAttr>();
        typeObject = dereference<pylir::Py::TypeAttr>(constant.getConstant());
        if (!typeObject)
        {
            return mlir::failure();
        }

        auto& map = typeObject.getSlots().getValue();
        auto iter = map.find("__slots__");
        if (iter == map.end())
        {
            rewriter.eraseOp(op);
            return mlir::success();
        }
        auto tupleAttr = dereference<pylir::Py::TupleAttr>(iter->second);
        PYLIR_ASSERT(tupleAttr);
        const auto* result = llvm::find_if(tupleAttr.getValue(),
                                           [&](mlir::Attribute attribute)
                                           {
                                               auto str = dereference<pylir::Py::StringAttr>(attribute);
                                               PYLIR_ASSERT(str);
                                               return str.getValueAttr() == adaptor.getSlot();
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
            instanceType = getBuiltinsInstanceType(getLayoutType(ref));
        }
        else
        {
            instanceType = getBuiltinsInstanceType(getLayoutType(typeObject));
        }
        mlir::Value i8Ptr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), adaptor.getObject().getType()), adaptor.getObject());
        i8Ptr = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), i8Ptr.getType(), i8Ptr, mlir::ValueRange{},
            llvm::ArrayRef<std::int32_t>{static_cast<std::int32_t>(sizeOf(instanceType))});
        auto objectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE), i8Ptr.getType()), i8Ptr);
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            op.getLoc(), objectPtrPtr.getType(), objectPtrPtr, mlir::ValueRange{},
            llvm::ArrayRef<std::int32_t>{static_cast<std::int32_t>(result - tupleAttr.getValue().begin())});
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(), gep);
        return mlir::success();
    }
};

struct GetSlotOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::GetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::GetSlotOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArgument(typeConverter->convertType(op.getType()), op.getLoc());

        rewriter.setInsertionPointToEnd(block);
        auto str = rewriter.create<pylir::Py::ConstantOp>(op.getLoc(),
                                                          pylir::Py::StringAttr::get(getContext(), adaptor.getSlot()));
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple =
            rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.getTypeObject(), typeRef, "__slots__");
        mlir::Value len = rewriter.create<pylir::Py::TupleLenOp>(op.getLoc(), rewriter.getIndexType(), slotsTuple);
        len = unrealizedConversion(rewriter, len);
        auto* condition = new mlir::Block;
        {
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getIndexType(), rewriter.getIndexAttr(0));
            condition->addArgument(getIndexType(), op.getLoc());
            rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{zero}, condition);
        }

        condition->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(condition);
        auto isLess = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ult,
                                                          condition->getArgument(0), len);
        auto unbound = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), endBlock->getArgument(0).getType());
        auto* body = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isLess, body, endBlock, mlir::ValueRange{unbound});

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto element = rewriter.create<pylir::Py::TupleGetItemOp>(op.getLoc(), slotsTuple, condition->getArgument(0));
        auto isEqual = rewriter.create<pylir::Py::StrEqualOp>(op.getLoc(), element, str);
        auto* foundIndex = new mlir::Block;
        auto* loop = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isEqual, foundIndex, loop);
        loop->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(loop);
        {
            auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getIndexType(), rewriter.getIndexAttr(1));
            auto increment = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), condition->getArgument(0), one);
            rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{increment}, condition);
        }

        foundIndex->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(foundIndex);
        auto typeObj = pyTypeModel(op.getLoc(), rewriter, adaptor.getTypeObject());
        auto offset = typeObj.offsetPtr(op.getLoc()).load(op.getLoc());
        auto index = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, condition->getArgument(0));
        auto pyObjectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE), adaptor.getObject().getType()),
            adaptor.getObject());
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), pyObjectPtrPtr.getType(), pyObjectPtrPtr,
                                                      mlir::ValueRange{index});
        auto slot = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), gep);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{slot}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        rewriter.replaceOp(op, endBlock->getArgument(0));
        return mlir::success();
    }
};

struct SetSlotOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::SetSlotOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::SetSlotOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});

        rewriter.setInsertionPointToEnd(block);
        auto str = rewriter.create<pylir::Py::ConstantOp>(op.getLoc(),
                                                          pylir::Py::StringAttr::get(getContext(), adaptor.getSlot()));
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple =
            rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.getTypeObject(), typeRef, "__slots__");
        mlir::Value len = rewriter.create<pylir::Py::TupleLenOp>(op.getLoc(), rewriter.getIndexType(), slotsTuple);
        len = unrealizedConversion(rewriter, len);
        auto* condition = new mlir::Block;
        {
            auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getIndexType(), rewriter.getIndexAttr(0));
            condition->addArgument(getIndexType(), op.getLoc());
            rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{zero}, condition);
        }

        condition->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(condition);
        auto isLess = rewriter.create<mlir::LLVM::ICmpOp>(op.getLoc(), mlir::LLVM::ICmpPredicate::ult,
                                                          condition->getArgument(0), len);
        auto* body = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isLess, body, endBlock);

        body->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(body);
        auto element = rewriter.create<pylir::Py::TupleGetItemOp>(op.getLoc(), slotsTuple, condition->getArgument(0));
        auto isEqual = rewriter.create<pylir::Py::StrEqualOp>(op.getLoc(), element, str);
        auto* foundIndex = new mlir::Block;
        auto* loop = new mlir::Block;
        rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), isEqual, foundIndex, loop);
        loop->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(loop);
        {
            auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getIndexType(), rewriter.getIndexAttr(1));
            auto increment = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), condition->getArgument(0), one);
            rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{increment}, condition);
        }

        foundIndex->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(foundIndex);
        auto typeObj = pyTypeModel(op.getLoc(), rewriter, adaptor.getTypeObject());
        auto offset = typeObj.offsetPtr(op.getLoc()).load(op.getLoc());
        auto index = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, condition->getArgument(0));
        auto pyObjectPtrPtr = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(pointer(getPyObjectType(), REF_ADDRESS_SPACE), adaptor.getObject().getType()),
            adaptor.getObject());
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), pyObjectPtrPtr.getType(), pyObjectPtrPtr,
                                                      mlir::ValueRange{index});
        rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), gep);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{}, endBlock);

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct RaiseOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::RaiseOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::RaiseOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::RaiseOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_raise, {adaptor.getException()});
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
            op, resultTypes, adaptor.getCalleeAttr(), adaptor.getCallOperands(), op.getHappyPath(),
            adaptor.getNormalDestOperands(), op.getExceptionPath(), adaptor.getUnwindDestOperands());
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
        llvm::SmallVector<mlir::Value> operands{adaptor.getCallee()};
        operands.append(adaptor.getCallOperands().begin(), adaptor.getCallOperands().end());
        rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(op, resultTypes, operands, op.getHappyPath(),
                                                          adaptor.getNormalDestOperands(), op.getExceptionPath(),
                                                          adaptor.getUnwindDestOperands());

        return mlir::success();
    }
};

struct LandingPadOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::LandingPadOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Py::LandingPadOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::LandingPadOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Value catchType;
        {
            mlir::OpBuilder::InsertionGuard guard{rewriter};
            rewriter.setInsertionPointToStart(&op->getParentRegion()->front());
            mlir::Value address = getConstant(op.getLoc(), adaptor.getCatchTypeAttr(), rewriter);
            while (auto cast = address.getDefiningOp<mlir::LLVM::BitcastOp>())
            {
                address = cast.getArg().getDefiningOp<mlir::LLVM::AddressOfOp>();
                PYLIR_ASSERT(address);
            }
            catchType = rewriter.create<mlir::LLVM::BitcastOp>(
                op.getLoc(), derivePointer(rewriter.getI8Type(), address.getType()), address);
        }
        // We use a integer of pointer width instead of a pointer to keep it opaque to statepoint passes.
        // Those do not support aggregates in aggregates.
        auto literal = mlir::LLVM::LLVMStructType::getLiteral(
            getContext(), {getIntPtrType(REF_ADDRESS_SPACE), rewriter.getI32Type()});
        auto landingPad = rewriter.create<mlir::LLVM::LandingpadOp>(op.getLoc(), literal, catchType);
        mlir::Value exceptionHeader = rewriter.create<mlir::LLVM::ExtractValueOp>(
            op.getLoc(), literal.getBody()[0], landingPad, rewriter.getI32ArrayAttr({0}));
        {
            // Itanium ABI mandates a pointer to the exception header be returned by the landing pad.
            // So we need to subtract the offset of the exception header inside of PyBaseException to get to it.
            auto pyBaseException = getPyBaseExceptionType();
            auto unwindHeader = getUnwindHeaderType();
            std::size_t offsetOf = 0;
            for (const auto& iter : pyBaseException.getBody())
            {
                offsetOf = llvm::alignTo(offsetOf, alignOf(iter));
                if (iter == unwindHeader)
                {
                    break;
                }
                offsetOf += sizeOf(iter);
            }
            auto byteOffset = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), exceptionHeader.getType(),
                                                                      rewriter.getI64IntegerAttr(offsetOf));
            exceptionHeader = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), exceptionHeader, byteOffset);
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, pointer(getPyObjectType(), REF_ADDRESS_SPACE),
                                                            exceptionHeader);
        return mlir::success();
    }
};

struct LandingPadBrOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::LandingPadBrOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::LandingPadBrOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, adaptor.getArguments(), op.getHandler());
        return mlir::success();
    }
};

struct GCAllocTupleConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocTupleOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocTupleOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::GCAllocTupleOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Value elementSize = createIndexConstant(rewriter, op.getLoc(), sizeOf(pointer(getPyObjectType())));
        elementSize = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), adaptor.getLength(), elementSize);
        mlir::Value headerSize = createIndexConstant(rewriter, op.getLoc(), sizeOf(getPyTupleType()));
        auto inBytes = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), elementSize, headerSize);
        auto memory = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto zeroI8 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
        auto falseC =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes, falseC);
        auto object =
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), memory);
        pyObjectModel(op.getLoc(), rewriter, object).typePtr(op.getLoc()).store(op.getLoc(), adaptor.getTypeObject());
        return mlir::success();
    }
};

struct GCAllocObjectConstTypeConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto constant = op.getTypeObject().getDefiningOp<pylir::Py::ConstantOp>();
        if (!constant)
        {
            return mlir::failure();
        }
        pylir::Py::TypeAttr typeAttr;
        auto ref = constant.getConstant().dyn_cast<mlir::FlatSymbolRefAttr>();
        typeAttr = dereference<pylir::Py::TypeAttr>(constant.getConstant());
        if (!typeAttr)
        {
            return mlir::failure();
        }

        std::size_t slotLen = 0;
        auto& map = typeAttr.getSlots().getValue();
        auto iter = map.find("__slots__");
        if (iter != map.end())
        {
            slotLen = dereference<pylir::Py::TupleAttr>(iter->second).getValue().size();
        }
        // I could create GEP here to read the offset component of the type object, but LLVM is not aware that the size
        // component is const, even if the rest of the type isn't. So instead we calculate the size here again to have
        // it be a constant.
        mlir::Type instanceType;
        if (ref)
        {
            instanceType = getBuiltinsInstanceType(getLayoutType(ref));
        }
        else
        {
            instanceType = getBuiltinsInstanceType(getLayoutType(typeAttr));
        }
        auto inBytes = createIndexConstant(rewriter, op.getLoc(),
                                           sizeOf(instanceType) + sizeOf(pointer(getPyObjectType())) * slotLen);
        auto memory = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto zeroI8 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
        auto falseC =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes, falseC);
        auto object =
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), memory);
        pyObjectModel(op.getLoc(), rewriter, object).typePtr(op.getLoc()).store(op.getLoc(), adaptor.getTypeObject());
        return mlir::success();
    }
};

struct GCAllocObjectOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::GCAllocObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::GCAllocObjectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto* block = op->getBlock();
        auto* endBlock = rewriter.splitBlock(block, mlir::Block::iterator{op});
        endBlock->addArgument(getIndexType(), op.getLoc());

        rewriter.setInsertionPointToEnd(block);
        auto typeRef = rewriter.create<pylir::Py::ConstantOp>(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Type.name));
        auto slotsTuple =
            rewriter.create<pylir::Py::GetSlotOp>(op.getLoc(), adaptor.getTypeObject(), typeRef, "__slots__");
        auto* hasSlotsBlock = new mlir::Block;
        {
            auto zero = createIndexConstant(rewriter, op.getLoc(), 0);
            auto hasNoSlots = rewriter.create<pylir::Py::IsUnboundValueOp>(op.getLoc(), slotsTuple);
            rewriter.create<mlir::LLVM::CondBrOp>(op.getLoc(), hasNoSlots, endBlock, mlir::ValueRange{zero},
                                                  hasSlotsBlock, mlir::ValueRange{});
        }

        hasSlotsBlock->insertBefore(endBlock);
        rewriter.setInsertionPointToStart(hasSlotsBlock);
        auto len = rewriter.create<pylir::Py::TupleLenOp>(op.getLoc(), getIndexType(), slotsTuple);
        rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), mlir::ValueRange{len}, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        auto offset =
            pyTypeModel(op.getLoc(), rewriter, adaptor.getTypeObject()).offsetPtr(op.getLoc()).load(op.getLoc());
        auto size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), offset, endBlock->getArgument(0));
        auto pointerSize = createIndexConstant(rewriter, op.getLoc(), sizeOf(pointer(getPyObjectType())));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, pointerSize);
        auto memory = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::pylir_gc_alloc, {inBytes});
        auto zeroI8 =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
        auto falseC =
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        rewriter.create<mlir::LLVM::MemsetOp>(op.getLoc(), memory, zeroI8, inBytes, falseC);
        auto object =
            rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, typeConverter->convertType(op.getType()), memory);
        pyObjectModel(op.getLoc(), rewriter, object).typePtr(op.getLoc()).store(op.getLoc(), adaptor.getTypeObject());
        return mlir::success();
    }
};

struct InitObjectOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitObjectOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitObjectOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitObjectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitTupleOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTupleOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTupleOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getMemory());
        auto size = createIndexConstant(rewriter, op.getLoc(), adaptor.getInitializer().size());

        tuple.sizePtr(op.getLoc()).store(op.getLoc(), size);
        auto trailing = tuple.trailingPtr(op.getLoc());
        for (const auto& iter : llvm::enumerate(adaptor.getInitializer()))
        {
            trailing.at(op.getLoc(), iter.index()).store(op.getLoc(), iter.value());
        }

        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitListOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitListOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitListOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto list = pyListModel(op.getLoc(), rewriter, adaptor.getMemory());
        auto size = createIndexConstant(rewriter, op.getLoc(), adaptor.getInitializer().size());

        list.sizePtr(op.getLoc()).store(op.getLoc(), size);

        auto tupleType = getConstant(
            op.getLoc(), mlir::FlatSymbolRefAttr::get(getContext(), pylir::Py::Builtins::Tuple.name), rewriter);
        auto tupleMemory = rewriter.create<pylir::Mem::GCAllocTupleOp>(op.getLoc(), tupleType, size);
        mlir::Value tupleInit =
            rewriter.create<pylir::Mem::InitTupleOp>(op.getLoc(), tupleMemory, adaptor.getInitializer());
        auto tuplePtr = list.tuplePtr(op.getLoc());
        tupleInit = unrealizedConversion(rewriter, tupleInit);
        tupleInit = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), mlir::Value{tuplePtr}.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType(),
            tupleInit);
        tuplePtr.store(op.getLoc(), tupleInit);

        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitTupleFromListOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTupleFromListOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTupleFromListOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTupleFromListOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getMemory());
        auto list = pyListModel(op.getLoc(), rewriter, adaptor.getInitializer());

        auto size = list.sizePtr(op.getLoc()).load(op.getLoc());
        tuple.sizePtr(op.getLoc()).store(op.getLoc(), size);

        auto sizeOf = createIndexConstant(rewriter, op.getLoc(), this->sizeOf(pointer(getPyObjectType())));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

        auto array = mlir::Value{tuple.trailingPtr(op.getLoc()).at(op.getLoc(), 0)};
        auto listArray =
            mlir::Value{list.tuplePtr(op.getLoc()).load(op.getLoc()).trailingPtr(op.getLoc()).at(op.getLoc(), 0)};
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), array.getType()), array);
        auto listArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), listArray.getType()), listArray);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, listArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));

        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitTuplePopFrontOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePopFrontOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePopFrontOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTuplePopFrontOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getMemory());
        auto prevTuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getTuple());

        mlir::Value size = prevTuple.sizePtr(op.getLoc()).load(op.getLoc());
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size, oneI);

        tuple.sizePtr(op.getLoc()).store(op.getLoc(), size);

        auto sizeOf = createIndexConstant(rewriter, op.getLoc(), this->sizeOf(pointer(getPyObjectType())));
        auto inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

        auto array = mlir::Value{tuple.trailingPtr(op.getLoc()).at(op.getLoc(), 0)};
        auto prevArray = mlir::Value{prevTuple.trailingPtr(op.getLoc()).at(op.getLoc(), 1)};
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), array.getType()), array);
        auto prevArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), prevArray.getType()), prevArray);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, prevArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitTuplePrependOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePrependOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitTuplePrependOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitTuplePrependOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto tuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getMemory());
        auto prevTuple = pyTupleModel(op.getLoc(), rewriter, adaptor.getTuple());

        mlir::Value size = prevTuple.sizePtr(op.getLoc()).load(op.getLoc());
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        auto resultSize = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, oneI);

        tuple.sizePtr(op.getLoc()).store(op.getLoc(), resultSize);
        tuple.trailingPtr(op.getLoc()).at(op.getLoc(), 0).store(op.getLoc(), adaptor.getElement());

        auto sizeOf = createIndexConstant(rewriter, op.getLoc(), this->sizeOf(pointer(getPyObjectType())));
        mlir::Value inBytes = rewriter.create<mlir::LLVM::MulOp>(op.getLoc(), size, sizeOf);

        auto array = mlir::Value{tuple.trailingPtr(op.getLoc()).at(op.getLoc(), 1)};
        auto prevArray = mlir::Value{prevTuple.trailingPtr(op.getLoc()).at(op.getLoc(), 0)};
        auto arrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), array.getType()), array);
        auto prevArrayI8 = rewriter.create<mlir::LLVM::BitcastOp>(
            op.getLoc(), derivePointer(rewriter.getI8Type(), prevArray.getType()), prevArray);
        rewriter.create<mlir::LLVM::MemcpyOp>(
            op.getLoc(), arrayI8, prevArrayI8, inBytes,
            rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false)));
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitIntOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitIntOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitIntOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto mpIntPointer = pyIntModel(op.getLoc(), rewriter, adaptor.getMemory()).mpIntPtr(op.getLoc());
        auto value = adaptor.getInitializer();
        if (value.getType() != rewriter.getI64Type())
        {
            value = rewriter.create<mlir::LLVM::ZExtOp>(op.getLoc(), rewriter.getI64Type(), value);
        }
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_init_u64,
                          {mlir::Value{mpIntPointer}, value});
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitStrOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitStrOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto string = pyStringModel(op.getLoc(), rewriter, adaptor.getMemory()).bufferPtr(op.getLoc());

        mlir::Value size = this->createIndexConstant(rewriter, op.getLoc(), 0);
        for (auto iter : adaptor.getStrings())
        {
            auto sizeLoaded = pyStringModel(op.getLoc(), rewriter, iter)
                                  .bufferPtr(op.getLoc())
                                  .sizePtr(op.getLoc())
                                  .load(op.getLoc());
            size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
        }

        string.sizePtr(op.getLoc()).store(op.getLoc(), size);
        string.capacityPtr(op.getLoc()).store(op.getLoc(), size);

        auto array = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::malloc, {size});
        string.elementPtr(op.getLoc()).store(op.getLoc(), array);

        size = this->createIndexConstant(rewriter, op.getLoc(), 0);
        for (auto iter : adaptor.getStrings())
        {
            auto iterString = pyStringModel(op.getLoc(), rewriter, iter).bufferPtr(op.getLoc());
            auto sizeLoaded = iterString.sizePtr(op.getLoc()).load(op.getLoc());
            auto sourceLoaded = mlir::Value{iterString.elementPtr(op.getLoc()).load(op.getLoc())};
            auto dest = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), array.getType(), array, size);
            auto falseC =
                rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
            rewriter.create<mlir::LLVM::MemcpyOp>(op.getLoc(), dest, sourceLoaded, sizeLoaded, falseC);
            size = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), size, sizeLoaded);
        }
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitStrFromIntOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrFromIntOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitStrFromIntOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitStrFromIntOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto string = pyStringModel(op.getLoc(), rewriter, adaptor.getMemory()).bufferPtr(op.getLoc());
        auto mpIntPtr = mlir::Value{pyIntModel(op.getLoc(), rewriter, adaptor.getInteger()).mpIntPtr(op.getLoc())};
        auto sizePtr = string.sizePtr(op.getLoc());
        auto ten = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), getInt(), rewriter.getI32IntegerAttr(10));
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_radix_size_overestimate,
                          {mpIntPtr, ten, mlir::Value{sizePtr}});
        auto capacity = sizePtr.load(op.getLoc());
        auto array = createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::malloc, {capacity});
        createRuntimeCall(op.getLoc(), rewriter, PylirTypeConverter::Runtime::mp_to_radix,
                          {mpIntPtr, array, capacity, mlir::Value{sizePtr}, ten});

        // mp_to_radix sadly includes the NULL terminator that it uses in size...
        mlir::Value size = sizePtr.load(op.getLoc());
        auto oneI = createIndexConstant(rewriter, op.getLoc(), 1);
        size = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), size, oneI);
        sizePtr.store(op.getLoc(), size);

        string.capacityPtr(op.getLoc()).store(op.getLoc(), capacity);
        string.elementPtr(op.getLoc()).store(op.getLoc(), array);

        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitFuncOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitFuncOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitFuncOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitFuncOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        auto address = rewriter.create<mlir::LLVM::AddressOfOp>(
            op.getLoc(), typeConverter->convertType(pylir::Py::getUniversalCCType(getContext())),
            adaptor.getInitializer());

        pyFunctionModel(op.getLoc(), rewriter, adaptor.getMemory()).funcPtr(op.getLoc()).store(op.getLoc(), address);
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct InitDictOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Mem::InitDictOp>
{
    using ConvertPylirOpToLLVMPattern<pylir::Mem::InitDictOp>::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Mem::InitDictOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getMemory());
        return mlir::success();
    }
};

struct ArithmeticSelectOpConversion : public ConvertPylirOpToLLVMPattern<mlir::arith::SelectOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        if (!op.getType().isa<pylir::Py::DynamicType, pylir::Mem::MemoryType>())
        {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(op, adaptor.getCondition(), adaptor.getTrueValue(),
                                                          adaptor.getFalseValue());
        return mlir::success();
    }
};

struct UnreachableOpConversion : public ConvertPylirOpToLLVMPattern<pylir::Py::UnreachableOp>
{
    using ConvertPylirOpToLLVMPattern::ConvertPylirOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(pylir::Py::UnreachableOp op, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
        return mlir::success();
    }
};

class ConvertPylirToLLVMPass : public pylir::ConvertPylirToLLVMBase<ConvertPylirToLLVMPass>
{
private:
    llvm::Triple m_triple;
    llvm::DataLayout m_dataLayout{""};

protected:
    void runOnOperation() override;

public:
    ConvertPylirToLLVMPass() = default;

    ConvertPylirToLLVMPass(llvm::Triple triple, const llvm::DataLayout& dataLayout)
        : m_triple(std::move(triple)), m_dataLayout(dataLayout)
    {
    }

    mlir::LogicalResult initializeOptions(llvm::StringRef options) override
    {
        if (mlir::failed(Pass::initializeOptions(options)))
        {
            return mlir::failure();
        }
        m_triple = llvm::Triple(m_targetTripleCLI.getValue());
        m_dataLayout = llvm::DataLayout(m_dataLayoutCLI.getValue());
        return mlir::success();
    }
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

    PylirTypeConverter converter(&getContext(), m_triple, m_dataLayout, module);
    converter.addConversion(
        [&](pylir::Py::DynamicType)
        { return mlir::LLVM::LLVMPointerType::get(converter.getPyObjectType(), REF_ADDRESS_SPACE); });
    converter.addConversion(
        [&](pylir::Mem::MemoryType)
        { return mlir::LLVM::LLVMPointerType::get(converter.getPyObjectType(), REF_ADDRESS_SPACE); });

    mlir::LLVMConversionTarget conversionTarget(getContext());
    conversionTarget.addIllegalDialect<pylir::Py::PylirPyDialect, pylir::Mem::PylirMemDialect>();
    conversionTarget.addLegalOp<mlir::ModuleOp>();

    mlir::RewritePatternSet patternSet(&getContext());
    mlir::populateFuncToLLVMConversionPatterns(converter, patternSet);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patternSet);
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
    patternSet.insert<SetSlotOpConstantConversion>(converter, 2);
    patternSet.insert<SetSlotOpConversion>(converter);
    patternSet.insert<StrEqualOpConversion>(converter);
    patternSet.insert<GCAllocTupleConversion>(converter);
    patternSet.insert<GCAllocObjectOpConversion>(converter);
    patternSet.insert<GCAllocObjectConstTypeConversion>(converter, 2);
    patternSet.insert<InitObjectOpConversion>(converter);
    patternSet.insert<InitListOpConversion>(converter);
    patternSet.insert<InitTupleOpConversion>(converter);
    patternSet.insert<InitTupleFromListOpConversion>(converter);
    patternSet.insert<ListAppendOpConversion>(converter);
    patternSet.insert<ListLenOpConversion>(converter);
    patternSet.insert<RaiseOpConversion>(converter);
    patternSet.insert<InitIntOpConversion>(converter);
    patternSet.insert<ObjectHashOpConversion>(converter);
    patternSet.insert<ObjectIdOpConversion>(converter);
    patternSet.insert<StrHashOpConversion>(converter);
    patternSet.insert<InitFuncOpConversion>(converter);
    patternSet.insert<InitDictOpConversion>(converter);
    patternSet.insert<DictTryGetItemOpConversion>(converter);
    patternSet.insert<DictSetItemOpConversion>(converter);
    patternSet.insert<DictDelItemOpConversion>(converter);
    patternSet.insert<DictLenOpConversion>(converter);
    patternSet.insert<InitStrOpConversion>(converter);
    patternSet.insert<PrintOpConversion>(converter);
    patternSet.insert<InitStrFromIntOpConversion>(converter);
    patternSet.insert<InvokeOpConversion>(converter);
    patternSet.insert<InvokeIndirectOpConversion>(converter);
    patternSet.insert<LandingPadOpConversion>(converter);
    patternSet.insert<BoolToI1OpConversion>(converter);
    patternSet.insert<InitTuplePrependOpConversion>(converter);
    patternSet.insert<InitTuplePopFrontOpConversion>(converter);
    patternSet.insert<IntGetIntegerOpConversion>(converter);
    patternSet.insert<IntCmpOpConversion>(converter);
    patternSet.insert<InitIntAddOpConversion>(converter);
    patternSet.insert<ArithmeticSelectOpConversion>(converter);
    patternSet.insert<LandingPadBrOpConversion>(converter);
    patternSet.insert<UnreachableOpConversion>(converter);
    patternSet.insert<TypeMROOpConversion>(converter);
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
        iter.setGarbageCollectorAttr(mlir::StringAttr::get(&getContext(), "pylir-gc"));
        iter.setPersonalityAttr(mlir::FlatSymbolRefAttr::get(&getContext(), "pylir_personality_function"));
    }
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                    mlir::StringAttr::get(&getContext(), m_dataLayout.getStringRepresentation()));
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    mlir::StringAttr::get(&getContext(), m_triple.str()));
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pylir::createConvertPylirToLLVMPass()
{
    return std::make_unique<ConvertPylirToLLVMPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    pylir::createConvertPylirToLLVMPass(llvm::Triple targetTriple, const llvm::DataLayout& dataLayout)
{
    return std::make_unique<ConvertPylirToLLVMPass>(std::move(targetTriple), dataLayout);
}
