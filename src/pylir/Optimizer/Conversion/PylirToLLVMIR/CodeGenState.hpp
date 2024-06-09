//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>

#include "PylirTypeConverter.hpp"

namespace pylir {
//===----------------------------------------------------------------------===//
// CodeGenBuilder
//===----------------------------------------------------------------------===//

/// Enum of possible TBAA Access types. With the exception of 'None', which is
/// used as a sentinel value to mean "Don't attach any TBAA information", the
/// values themselves don't actually have any meaning attached to them. Rather
/// they're all siblings in the TBAA hierarchy and therefore can only alias
/// memory accesses with the same or no TBAA information.
enum class TbaaAccessType {
  None,
  Slots,
  TypeObject,
  TupleElements,
  TupleSize,
  ListTupleMember,
  ListSize,
  DictSize,
  StringSize,
  StringCapacity,
  StringElementPtr,
  FloatValue,
  FunctionPointer,
  TypeMroMember,
  TypeSlotsMember,
  TypeOffset,
  Handle,
};

/// Class for managing everything module level during code generation. This
/// includes code generation for globals, generating function declarations and
/// more.
class CodeGenState {
  mlir::LLVM::LLVMPointerType m_objectPtrType;
  PylirTypeConverter& m_typeConverter;
  mlir::SymbolTable m_symbolTable;

  /// Mapping of all `ConcreteObjectAttribute` to the corresponding converted
  /// LLVM global.
  llvm::DenseMap<Py::ConcreteObjectAttribute, mlir::LLVM::GlobalOp>
      m_constantObjects;
  /// Mapping for any auxiliary buffers used by a `ConcreteObjectAttribute`.
  /// The attribute to use as key is specific to the implementation. Buffers
  /// can be reused by making equivalent data use the same key when converting
  /// a given object attribute.
  llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> m_globalBuffers;
  /// Mapping of any `GlobalValueAttr` to the corresponding converted LLVM
  /// global.
  llvm::DenseMap<Py::GlobalValueAttr, mlir::LLVM::GlobalOp> m_globalValues;
  /// Mapping of any `GlobalValueAttr` that occur directly in a `py.external`
  /// operation to the symbol name of that operation.
  llvm::DenseMap<Py::GlobalValueAttr, mlir::StringAttr> m_externalGlobalValues;
  mlir::LLVM::LLVMFuncOp m_globalInit;
  mlir::LLVM::TBAARootAttr m_tbaaRoot;

  void appendToGlobalInit(mlir::OpBuilder& builder,
                          llvm::function_ref<void()> section);

  /// Gets or converts the given `objectAttr` to an LLVM global.
  mlir::LLVM::GlobalOp
  getConstantObject(mlir::OpBuilder& builder,
                    Py::ConcreteObjectAttribute objectAttr);

  /// Generates code to create the initializer region for 'global' with the
  /// compile time constant 'objectAttr'.
  void initializeGlobal(mlir::LLVM::GlobalOp global, mlir::OpBuilder& builder,
                        Py::ConcreteObjectAttribute objectAttr);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::StrAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::TupleAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::ListAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::FloatAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::IntAttrInterface attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::DictAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::TypeAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

  mlir::Value initialize(mlir::Location loc, mlir::OpBuilder& builder,
                         Py::FunctionAttr attr, mlir::Value undef,
                         mlir::LLVM::GlobalOp global);

public:
  CodeGenState(PylirTypeConverter& typeConverter, mlir::ModuleOp module)
      : m_objectPtrType(mlir::LLVM::LLVMPointerType::get(
            &typeConverter.getContext(), REF_ADDRESS_SPACE)),
        m_typeConverter(typeConverter), m_symbolTable(module),
        m_tbaaRoot(mlir::LLVM::TBAARootAttr::get(
            &typeConverter.getContext(),
            mlir::StringAttr::get(&typeConverter.getContext(),
                                  "Pylir TBAA Root"))) {
    for (auto exportOp : module.getOps<Py::ExternalOp>())
      m_externalGlobalValues.insert(
          {exportOp.getAttr(), exportOp.getSymNameAttr()});
  }

  /// Returns an array of symbol references for the given 'TbaaAccessType',
  /// suitable for directly attaching to load or store instructions.
  mlir::ArrayAttr getTBAAAccess(TbaaAccessType accessType);

  /// Enum of runtime functions called by the generated code. These are all
  /// either libc functions or functions implemented in Pylir's runtime.
  enum class Runtime {
    // NOLINTBEGIN(readability-identifier-naming): For clarity purpose, these
    // enum values should have the exact same case as the actual functions
    // they're representing.
    memcmp,
    malloc,
    mp_init_u64,
    mp_init_i64,
    mp_get_i64,
    mp_init,
    mp_unpack,
    mp_radix_size_overestimate,
    mp_to_radix,
    mp_cmp,
    mp_add,
    pylir_gc_alloc,
    pylir_str_hash,
    pylir_dict_lookup,
    pylir_dict_insert,
    pylir_dict_insert_unique,
    pylir_dict_erase,
    pylir_print,
    pylir_raise,
    // NOLINTEND(readability-identifier-naming)
  };

  /// Creates a call to the runtime function indicated by 'func'. 'args' are the
  /// arguments of the call. Returns the return value of the function, if it has
  /// one.
  mlir::Value createRuntimeCall(mlir::Location loc, mlir::OpBuilder& builder,
                                Runtime func, mlir::ValueRange args);

  /// Generates code to translate the compile time constant 'attribute' to an
  /// PyObject pointer in LLVM and returns it. Attribute may be any kind of
  /// attribute from the 'py' dialect.
  mlir::Value getConstant(mlir::Location loc, mlir::OpBuilder& builder,
                          mlir::Attribute attribute);

  /// Gets or lowers 'globalValueAttr' to a LLVM global variable using
  /// 'builder'. Once lowered, the same global is returned for the same
  /// 'globalValueAttr'.
  mlir::LLVM::GlobalOp getGlobalValue(mlir::OpBuilder& builder,
                                      Py::GlobalValueAttr globalValueAttr);

  /// Returns the type converter used.
  PylirTypeConverter& getTypeConverter() const {
    return m_typeConverter;
  }

  /// Returns the global initializer function. This function is created and
  /// inserted into lazily when encountering an attribute requiring runtime
  /// initialization. If no such attribute was ever encountered, a null value is
  /// returned.
  mlir::LLVM::LLVMFuncOp getGlobalInit() const {
    return m_globalInit;
  }
};

//===----------------------------------------------------------------------===//
// Models
//===----------------------------------------------------------------------===//

/// Base class and base case used to nicely model and chain code generation of
/// access into Python objects. It offers basic functionality and interfaces
/// shared across all model implementations. New models are simply created by
/// subclassing models. Other models must make sure to inherit the constructor
/// of 'Model'.
///
/// There are two kinds of models: Generic models and Concrete models. The
/// former is a model that operates on a type which is parameterized in some
/// way. Concrete models on the other hand, only work on a singular type, which
/// they return via the static 'getElementType' method. This type is also always
/// used on construction.
///
/// Important to note is that models ALWAYS operate on *pointer*s to the
/// modelled type. This is required due to how LLVM works, since plain SSA
/// values do not have storage or an address, making things like random access
/// impossible. Having it in memory, usually via memory allocated by the GC or
/// through stack allocation ('alloca'), we just need to use GEPs, loads and
/// stores. LLVM optimizes out redundant loads, stores and stack allocation very
/// well anyways.
template <class ConcreteModel, class Type>
struct Model {
  template <class T, class = void>
  struct HasConcreteElementType : std::false_type {};

  template <class T>
  struct HasConcreteElementType<T, std::void_t<decltype(T::getElementType(
                                       std::declval<PylirTypeConverter&>()))>>
      : std::true_type {};

  static mlir::Type getElementTypeOr(PylirTypeConverter& typeConverter,
                                     mlir::Type other) {
    if constexpr (HasConcreteElementType<ConcreteModel>{})
      return ConcreteModel::getElementType(typeConverter);
    else
      return other;
  }

protected:
  mlir::OpBuilder& m_builder;
  mlir::Value m_pointer;
  Type m_elementType;
  CodeGenState& m_codeGenState;

public:
  /// Common signature to construct any 'Model' or subclass. It consists of the
  /// builder that models will use to generate code for their accessors, the
  /// actual 'pointer' value, whose element type is 'elementType' and a type
  /// converter often used to get result types. The 'elementType' is essential
  /// for generic models and overwritten with the actual type for concrete
  /// models.
  Model(mlir::OpBuilder& builder, mlir::Value pointer, mlir::Type elementType,
        CodeGenState& codeGenState)
      : m_builder(builder), m_pointer(pointer),
        m_elementType(mlir::cast<Type>(
            getElementTypeOr(codeGenState.getTypeConverter(), elementType))),
        m_codeGenState(codeGenState) {}

  /// Convenience constructor for concrete models that do not require specifying
  /// an element type.
  template <class U = ConcreteModel,
            std::enable_if_t<HasConcreteElementType<U>{}>* = nullptr>
  Model(mlir::OpBuilder& builder, mlir::Value pointer,
        CodeGenState& codeGenState)
      : m_builder(builder), m_pointer(pointer),
        m_elementType(U::getElementType(codeGenState.getTypeConverter())),
        m_codeGenState(codeGenState) {}

  /// Implicit conversion to the contained pointer value.
  /*implicit*/ operator mlir::Value() const {
    return m_pointer;
  }

  /// Implicit conversion to the contained pointer value.
  /// Conversion to 'ValueRange' is also added here since a lot of Op builders
  /// take 'ValueRange' and C++ cannot do two implicit conversions in
  /// initialization.
  /*implicit*/ operator mlir::ValueRange() const {
    return m_pointer;
  }

  constexpr static bool isConcreteModel() {
    return HasConcreteElementType<ConcreteModel>{};
  }
};

/// Generic model for simple scalar values. Can be used with any actual element
/// type and only supports loading and storing values. An optional template
/// parameter allows defining the TBAA access type that should be used for load
/// and store instructions.
template <TbaaAccessType tbaaAccessType = TbaaAccessType::None>
struct Scalar : Model<Scalar<tbaaAccessType>, mlir::Type> {
  using Model<Scalar<tbaaAccessType>, mlir::Type>::Model;

  /// Generates a load access for the scalar, returning the actual scalar value.
  mlir::Value load(mlir::Location loc) const {
    mlir::LLVM::LoadOp loadOp =
        this->m_builder.template create<mlir::LLVM::LoadOp>(
            loc, this->m_elementType, this->m_pointer);
    loadOp.setTbaaAttr(this->m_codeGenState.getTBAAAccess(tbaaAccessType));
    return loadOp;
  }

  /// Stores a value into the memory pointed to by the internal pointer.
  mlir::LLVM::StoreOp store(mlir::Location loc, mlir::Value value) const {
    auto store = this->m_builder.template create<mlir::LLVM::StoreOp>(
        loc, value, this->m_pointer);
    store.setTbaaAttr(this->m_codeGenState.getTBAAAccess(tbaaAccessType));
    return store;
  }
};

/// Base class for models over LLVM struct types. Allows accessing the fields of
/// the struct using a convenient method.
template <class ConcreteModel>
struct StructModelBase : Model<ConcreteModel, mlir::LLVM::LLVMStructType> {
  using Model<ConcreteModel, mlir::LLVM::LLVMStructType>::Model;

  /// Method for accessing the fields of a struct. 'ResultModel' is the return
  /// type and should be seen as the type of the field. The field accessed is
  /// the one specified by 'index' and is simply the index of the field within
  /// the LLVM struct type.
  template <class ResultModel>
  ResultModel field(mlir::Location loc, std::size_t index) const {
    return {this->m_builder,
            this->m_builder.template create<mlir::LLVM::GEPOp>(
                loc, this->m_pointer.getType(), this->m_elementType,
                this->m_pointer,
                llvm::ArrayRef<mlir::LLVM::GEPArg>{
                    0, static_cast<std::int32_t>(index)}),
            this->m_elementType.getBody()[index], this->m_codeGenState};
  }
};

/// Generic Model for Pointers to another model. Requires 'ElementModel' to be a
/// concrete model, since LLVM pointers are opaque and do not have any knowledge
/// about their element type. An optional template parameter allows defining the
/// TBAA access type that should be used for load and store instructions.
template <class ElementModel,
          TbaaAccessType tbaaAccessType = TbaaAccessType::None>
struct Pointer : Model<Pointer<ElementModel, tbaaAccessType>,
                       mlir::LLVM::LLVMPointerType> {
  using Model<Pointer<ElementModel, tbaaAccessType>,
              mlir::LLVM::LLVMPointerType>::Model;

  static_assert(ElementModel::isConcreteModel(),
                "Pointer model only works on concrete models");

  /// Loads the pointer, returning an instance of the element model.
  ElementModel load(mlir::Location loc) const {
    mlir::LLVM::LoadOp loadOp =
        this->m_builder.template create<mlir::LLVM::LoadOp>(
            loc, this->m_elementType, this->m_pointer);
    loadOp.setTbaaAttr(this->m_codeGenState.getTBAAAccess(tbaaAccessType));
    return {this->m_builder, loadOp, this->m_codeGenState};
  }

  /// Stores a given value into the contained pointer.
  mlir::LLVM::StoreOp store(mlir::Location loc, mlir::Value value) const {
    auto store = this->m_builder.template create<mlir::LLVM::StoreOp>(
        loc, value, this->m_pointer);
    store.setTbaaAttr(this->m_codeGenState.getTBAAAccess(tbaaAccessType));
    return store;
  }

  /// Returns a new pointer model with a runtime offset applied. The offset is
  /// interpreted similar to using '+' in C, that is, an offset of one, moves to
  /// the next element in an array of the given element type. In other words, it
  /// advances the pointer value by 'index' * 'sizeof elementType bytes.
  Pointer<ElementModel, tbaaAccessType> offset(mlir::Location loc,
                                               mlir::Value index) const {
    return {this->m_builder,
            this->m_builder.template create<mlir::LLVM::GEPOp>(
                loc, this->m_pointer.getType(), this->m_elementType,
                this->m_pointer, index),
            this->m_codeGenState};
  }

  /// Convenience overload of the above with a compile time constant offset.
  Pointer<ElementModel, tbaaAccessType> offset(mlir::Location loc,
                                               std::int32_t index) const {
    return {this->m_builder,
            this->m_builder.template create<mlir::LLVM::GEPOp>(
                loc, this->m_pointer.getType(), this->m_elementType,
                this->m_pointer, llvm::ArrayRef<mlir::LLVM::GEPArg>{index}),
            this->m_codeGenState};
  }
};

/// Generic Model for arrays of a given model. Allows convenient access to each
/// element in the array.
template <class ElementModel>
struct Array : Model<Array<ElementModel>, mlir::LLVM::LLVMArrayType> {
  using Model<Array<ElementModel>, mlir::LLVM::LLVMArrayType>::Model;

  /// Returns a model instance of the element pointed to by the given runtime
  /// 'index'.
  ElementModel at(mlir::Location loc, mlir::Value index) const {
    return {this->m_builder,
            this->m_builder.template create<mlir::LLVM::GEPOp>(
                loc, this->m_pointer.getType(), this->m_elementType,
                this->m_pointer, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, index}),
            this->m_elementType.getElementType(), this->m_codeGenState};
  }

  /// Same as above but the index is a runtime constant.
  ElementModel at(mlir::Location loc, std::int32_t index) const {
    return {this->m_builder,
            this->m_builder.template create<mlir::LLVM::GEPOp>(
                loc, this->m_pointer.getType(), this->m_elementType,
                this->m_pointer, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, index}),
            this->m_elementType.getElementType(), this->m_codeGenState};
  }

  template <class T>
  std::enable_if_t<std::is_enum_v<T>, ElementModel> at(mlir::Location loc,
                                                       T index) const {
    return at(loc, static_cast<std::underlying_type_t<T>>(index));
  }
};

struct PyTypeModel;

/// Base class for any models of structs effectively inheriting from 'PyObject'.
template <class ConcreteOp>
struct PyObjectModelBase : StructModelBase<ConcreteOp> {
  using StructModelBase<ConcreteOp>::StructModelBase;

  /// Returns a model for the pointer to the type object.
  auto typePtr(mlir::Location loc) const {
    return this
        ->template field<Pointer<PyTypeModel, TbaaAccessType::TypeObject>>(loc,
                                                                           0);
  }
};

/// Concrete model for 'PyObject'.
struct PyObjectModel : PyObjectModelBase<PyObjectModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyObjectType();
  }
};

/// Concrete model for 'PyTuple'.
struct PyTupleModel : PyObjectModelBase<PyTupleModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyTupleType();
  }

  /// Returns a model for accessing the size of the tuple.
  auto size(mlir::Location loc) const {
    return field<Scalar<TbaaAccessType::TupleSize>>(loc, 1);
  }

  /// Returns a model for accessing the trailing objects, or in other words, the
  /// elements within the tuple.
  auto trailingArray(mlir::Location loc) const {
    return field<Array<Pointer<PyObjectModel, TbaaAccessType::TupleElements>>>(
        loc, 2);
  }
};

/// Concrete model for 'PyList'.
struct PyListModel : PyObjectModelBase<PyListModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyListType();
  }

  /// Returns a model for accessing the size of the list.
  auto size(mlir::Location loc) const {
    return field<Scalar<TbaaAccessType::ListSize>>(loc, 1);
  }

  /// Returns a model for accessing the reference to the internal tuple.
  /// The internal tuple is the actual storage of the list and the size of the
  /// tuple is the capacity of the list.
  auto tuplePtr(mlir::Location loc) const {
    return field<Pointer<PyTupleModel, TbaaAccessType::ListTupleMember>>(loc,
                                                                         2);
  }
};

/// Concrete model for 'BufferComponent'. Currently used by 'PyString' and
/// 'PyDict' as backing storage. Optional template parameters allow defining the
/// TBAA access types of store instructions for size, capacity and the element
/// pointer.
template <TbaaAccessType sizeAccess = TbaaAccessType::None,
          TbaaAccessType capacityAccess = TbaaAccessType::None,
          TbaaAccessType elementAccess = TbaaAccessType::None>
struct BufferComponentModel
    : StructModelBase<
          BufferComponentModel<sizeAccess, capacityAccess, elementAccess>> {
  using StructModelBase<BufferComponentModel<sizeAccess, capacityAccess,
                                             elementAccess>>::StructModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getBufferComponent();
  }

  /// Returns a model for accessing the size of the buffer.
  auto size(mlir::Location loc) const {
    return this->template field<Scalar<sizeAccess>>(loc, 0);
  }

  /// Returns a model for accessing the capacity of the buffer.
  auto capacity(mlir::Location loc) const {
    return this->template field<Scalar<capacityAccess>>(loc, 1);
  }

  /// Returns a model for accessing the pointer to the allocated storage.
  /// Note that this returns a 'Scalar' Model, as the element type of the
  /// pointer is unknown and up to the user.
  auto elementPtr(mlir::Location loc) const {
    return this->template field<Scalar<elementAccess>>(loc, 2);
  }
};

/// Concrete model for 'PyDict'.
struct PyDictModel : PyObjectModelBase<PyDictModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyDictType();
  }

  /// Returns a model for the internal key-value pair list of the dictionary.
  auto bufferPtr(mlir::Location loc) const {
    return field<BufferComponentModel<TbaaAccessType::DictSize>>(loc, 1);
  }
};

/// Concrete model for 'mp_int', the arbitrary sized integer from libtommath.
struct MPIntModel : StructModelBase<MPIntModel> {
  using StructModelBase::StructModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getMPInt();
  }

  /// Returns a model for the internal 'used' member, describing how many digits
  /// are currently in use.
  auto used(mlir::Location loc) const {
    return field<Scalar<>>(loc, 0);
  }
};

/// Concrete model for 'PyInt'.
struct PyIntModel : PyObjectModelBase<PyIntModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyIntType();
  }

  /// Returns a model for the wrapped 'mp_int'.
  auto mpInt(mlir::Location loc) const {
    return field<MPIntModel>(loc, 1);
  }
};

/// Concrete model for 'PyFloat'.
struct PyFloatModel : PyObjectModelBase<PyFloatModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyFloatType();
  }

  /// Returns a model for the wrapped double value.
  auto doubleValue(mlir::Location loc) const {
    return field<Scalar<TbaaAccessType::FloatValue>>(loc, 1);
  }
};

/// Concrete model for 'PyFunction'.
struct PyFunctionModel : PyObjectModelBase<PyFunctionModel> {
  using PyObjectModelBase::PyObjectModelBase;

  /// Returns a model for the internal function pointer, using the universal
  /// calling convention.
  auto funcPtr(mlir::Location loc) const {
    return field<Scalar<TbaaAccessType::FunctionPointer>>(loc, 1);
  }

  // Returns a model for accessing the field denoting the size of all closure
  // arguments combined as laid out in memory.
  auto closureSizePtr(mlir::Location loc) const {
    return field<Scalar<>>(loc, 2);
  }

  /// Returns a model for accessing the slots of the function.
  auto slotsArray(mlir::Location loc) const {
    return field<Array<Pointer<PyObjectModel, TbaaAccessType::TupleElements>>>(
        loc, 3);
  }

  /// Returns a model for accessing the closure argument with the given index.
  auto closureArgument(mlir::Location loc, unsigned index) const {
    return field<Scalar<>>(loc, 4 + index);
  }

  auto refInClosureBitfield(mlir::Location loc, unsigned numClosureArgs) const {
    return field<Array<Scalar<>>>(loc, 4 + numClosureArgs);
  }
};

/// Concrete model for 'PyString'.
struct PyStringModel : PyObjectModelBase<PyStringModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyStringType();
  }

  /// Returns a model for the buffer component used to store the characters in
  /// UTF-8.
  auto buffer(mlir::Location loc) const {
    return field<BufferComponentModel<TbaaAccessType::StringSize,
                                      TbaaAccessType::StringCapacity,
                                      TbaaAccessType::StringElementPtr>>(loc,
                                                                         1);
  }
};

/// Concrete model for 'PyType'.
struct PyTypeModel : PyObjectModelBase<PyTypeModel> {
  using PyObjectModelBase::PyObjectModelBase;

  static auto getElementType(PylirTypeConverter& typeConverter) {
    return typeConverter.getPyTypeType();
  }

  /// Returns a model for the internal offset member.
  /// The offset is essentially the size of the struct used for instances of
  /// this type object minus any padding at the end of the object. Its units is
  /// in pointer sizes and it is effectively used as an offset when accessing
  /// slots of an object.
  auto offset(mlir::Location loc) const {
    return field<Scalar<TbaaAccessType::TypeOffset>>(loc, 1);
  }

  /// Returns a model for the layout type. The layout type is the base class
  /// that has determined the memory layout of an instance of this type object.
  /// E.g. all subclasses of 'int' have 'int' as their layout type. A type can
  /// only have one layout type.
  auto layoutPtr(mlir::Location loc) const {
    return field<Pointer<PyTypeModel>>(loc, 2);
  }

  /// Returns a model for the pointer to the MRO tuple. The MRO tuple, or
  /// "method resolution order" tuple is simply the tuple of all base classes of
  /// the type in the order in which method lookup should search base classes to
  /// find a method. First entry is always the type object itself.
  auto mroPtr(mlir::Location loc) const {
    return field<Pointer<PyTupleModel, TbaaAccessType::TypeMroMember>>(loc, 3);
  }

  /// Returns a model for the pointer to the tuple of slots. This is simply a
  /// tuple of strings where each string is the name of the slot of an instance
  /// of this type with the corresponding index. The size of this tuple is
  /// therefore also equal to the amount of slots an instance of this type.
  auto instanceSlotsPtr(mlir::Location loc) const {
    return field<Pointer<PyTupleModel, TbaaAccessType::TypeSlotsMember>>(loc,
                                                                         4);
  }

  /// Returns a model for accessing the slots of the function.
  auto slotsArray(mlir::Location loc) const {
    return field<Array<Pointer<PyObjectModel, TbaaAccessType::TupleElements>>>(
        loc, 5);
  }
};

inline bool needToBeRuntimeInit(Py::ObjectAttrInterface attr) {
  // Integer attrs currently need to be runtime init due to memory allocation in
  // libtommath Dict attr need to be runtime init due to the hash calculation
  return mlir::isa<Py::IntAttr, Py::BoolAttr, Py::DictAttr>(attr);
}

} // namespace pylir
