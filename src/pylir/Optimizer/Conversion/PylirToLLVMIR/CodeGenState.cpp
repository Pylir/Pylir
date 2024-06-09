//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGenState.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirMem/IR/Value.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

void CodeGenState::appendToGlobalInit(OpBuilder& builder,
                                      llvm::function_ref<void()> section) {
  OpBuilder::InsertionGuard guard{builder};
  if (!m_globalInit) {
    builder.setInsertionPointToEnd(
        cast<ModuleOp>(m_symbolTable.getOp()).getBody());
    m_globalInit = builder.create<LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "$__GLOBAL_INIT__",
        LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(), {}),
        LLVM::Linkage::Internal, true);
    m_globalInit.addEntryBlock(builder);
  }
  builder.setInsertionPointToEnd(&m_globalInit.back());
  section();
}

ArrayAttr CodeGenState::getTBAAAccess(TbaaAccessType accessType) {
  MLIRContext* context = &m_typeConverter.getContext();
  if (accessType == TbaaAccessType::None)
    return ArrayAttr::get(context, {});

  std::string tbaaAccessTypeString;
  switch (accessType) {
  case TbaaAccessType::None: PYLIR_UNREACHABLE;
  case TbaaAccessType::Slots:
    tbaaAccessTypeString = "Python Object Slots";
    break;
  case TbaaAccessType::TypeObject:
    tbaaAccessTypeString = "Python Type Object";
    break;
  case TbaaAccessType::TupleElements:
    tbaaAccessTypeString = "Python Tuple Elements";
    break;
  case TbaaAccessType::ListTupleMember:
    tbaaAccessTypeString = "Python List Tuple";
    break;
  case TbaaAccessType::TypeMroMember:
    tbaaAccessTypeString = "Python Type MRO";
    break;
  case TbaaAccessType::TypeSlotsMember:
    tbaaAccessTypeString = "Python Type Instance Slots";
    break;
  case TbaaAccessType::TypeOffset:
    tbaaAccessTypeString = "Python Type Offset";
    break;
  case TbaaAccessType::Handle: tbaaAccessTypeString = "Python Handle"; break;
  case TbaaAccessType::TupleSize:
    tbaaAccessTypeString = "Python Tuple Size";
    break;
  case TbaaAccessType::ListSize:
    tbaaAccessTypeString = "Python List Size";
    break;
  case TbaaAccessType::DictSize:
    tbaaAccessTypeString = "Python Dict Size";
    break;
  case TbaaAccessType::StringSize:
    tbaaAccessTypeString = "Python String Size";
    break;
  case TbaaAccessType::StringCapacity:
    tbaaAccessTypeString = "Python String Capacity";
    break;
  case TbaaAccessType::StringElementPtr:
    tbaaAccessTypeString = "Python String Element Ptr";
    break;
  case TbaaAccessType::FloatValue:
    tbaaAccessTypeString = "Python Float Value";
    break;
  case TbaaAccessType::FunctionPointer:
    tbaaAccessTypeString = "Python Function Pointer";
    break;
  }

  auto type = LLVM::TBAATypeDescriptorAttr::get(
      context, /*identity=*/tbaaAccessTypeString,
      LLVM::TBAAMemberAttr::get(m_tbaaRoot, 0));

  auto tag =
      LLVM::TBAATagAttr::get(/*base_type=*/type, /*access_type=*/type, 0);

  return ArrayAttr::get(context, tag);
}

Value CodeGenState::createRuntimeCall(Location loc, OpBuilder& builder,
                                      CodeGenState::Runtime func,
                                      ValueRange args) {
  PlatformABI& abi = m_typeConverter.getPlatformABI();
  MLIRContext* context = builder.getContext();
  auto pointerType = builder.getType<LLVM::LLVMPointerType>();

  Type returnType;
  NamedAttrList resultAttrs;
  llvm::SmallVector<Type> argumentTypes;
  std::string functionName;
  llvm::SmallVector<llvm::StringRef> passThroughAttributes;
  switch (func) {
  case Runtime::memcmp:
    returnType = abi.getInt(context);
    argumentTypes = {pointerType, pointerType, m_typeConverter.getIndexType()};
    functionName = "memcmp";
    break;
  case Runtime::malloc:
    returnType = pointerType;
    argumentTypes = {m_typeConverter.getIndexType()};
    functionName = "malloc";
    break;
  case Runtime::pylir_gc_alloc:
    returnType = m_objectPtrType;
    argumentTypes = {m_typeConverter.getIndexType()};
    functionName = "pylir_gc_alloc";
    // TODO: Set allockind("alloc,zeroed") allocsize(0) LLVM attributes once
    // supported upstream.
    break;
  case Runtime::mp_init_u64:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, builder.getI64Type()};
    functionName = "mp_init_u64";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_init_i64:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, builder.getI64Type()};
    functionName = "mp_init_i64";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_get_i64:
    returnType = builder.getI64Type();
    argumentTypes = {m_objectPtrType};
    functionName = "mp_get_i64";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::pylir_str_hash:
    returnType = m_typeConverter.getIndexType();
    argumentTypes = {m_objectPtrType};
    functionName = "pylir_str_hash";
    passThroughAttributes = {"nounwind"};
    break;
  case Runtime::pylir_print:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType};
    functionName = "pylir_print";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::pylir_raise:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType};
    functionName = "pylir_raise";
    passThroughAttributes = {"noreturn"};
    break;
  case Runtime::mp_init:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType};
    functionName = "mp_init";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_unpack:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType,     m_typeConverter.getIndexType(),
                     abi.getInt(context), m_typeConverter.getIndexType(),
                     abi.getInt(context), m_typeConverter.getIndexType(),
                     pointerType};
    functionName = "mp_unpack";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_radix_size_overestimate:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, abi.getInt(context), m_objectPtrType};
    functionName = "mp_radix_size_overestimate";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_to_radix:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, pointerType,
                     m_typeConverter.getIndexType(), m_objectPtrType,
                     abi.getInt(context)};
    functionName = "mp_to_radix";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_cmp:
    returnType = abi.getInt(context);
    argumentTypes = {m_objectPtrType, m_objectPtrType};
    functionName = "mp_cmp";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::mp_add:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, m_objectPtrType, m_objectPtrType};
    functionName = "mp_add";
    passThroughAttributes = {"gc-leaf-function", "nounwind"};
    break;
  case Runtime::pylir_dict_lookup:
    returnType = m_objectPtrType;
    argumentTypes = {m_objectPtrType, returnType, abi.getSizeT(context)};
    functionName = "pylir_dict_lookup";
    break;
  case Runtime::pylir_dict_erase:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context)};
    functionName = "pylir_dict_erase";
    break;
  case Runtime::pylir_dict_insert:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context),
                     m_objectPtrType};
    functionName = "pylir_dict_insert";
    break;
  case Runtime::pylir_dict_insert_unique:
    returnType = LLVM::LLVMVoidType::get(context);
    argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context),
                     m_objectPtrType};
    functionName = "pylir_dict_insert_unique";
    break;
  }

  auto module = cast<ModuleOp>(m_symbolTable.getOp());
  auto llvmFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(functionName);
  if (!llvmFunc) {
    OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToEnd(module.getBody());
    llvmFunc =
        abi.declareFunc(builder, loc, returnType, functionName, argumentTypes);
    if (!passThroughAttributes.empty())
      llvmFunc.setPassthroughAttr(
          builder.getStrArrayAttr(passThroughAttributes));

    if (!resultAttrs.empty())
      llvmFunc.setResultAttrs(0, resultAttrs);
  }
  return abi.callFunc(builder, loc, llvmFunc, args);
}

void CodeGenState::initializeGlobal(LLVM::GlobalOp global, OpBuilder& builder,
                                    ConcreteObjectAttribute objectAttr) {
  builder.setInsertionPointToStart(
      &global.getInitializerRegion().emplaceBlock());
  Value undef =
      builder.create<LLVM::UndefOp>(global.getLoc(), global.getType());
  ConstObjectAttrInterface constObjectAttrInterface = objectAttr;
  ObjectBaseAttribute typeObject = constObjectAttrInterface.getTypeObject();

  Value typeObj = getConstant(global.getLoc(), builder, typeObject);
  undef =
      builder.create<LLVM::InsertValueOp>(global.getLoc(), undef, typeObj, 0);
  undef =
      llvm::TypeSwitch<ConcreteObjectAttribute, Value>(objectAttr)
          .Case<StrAttr, TupleAttr, ListAttr, Py::FloatAttr, IntAttrInterface,
                DictAttr, Py::TypeAttr, FunctionAttr>([&](auto attr) {
            return initialize(global.getLoc(), builder, attr, undef, global);
          })
          .Case([&](ObjectAttr) { return undef; });

  unsigned slotsInStructIndex =
      cast<LLVM::LLVMStructType>(global.getType()).getBody().size() - 1;
  // Special case due to functions not having slots as the second last struct
  // element.
  // TODO: This will need to be calculated from the 'FunctionAttr' once/if it is
  // ever capable of representing closure arguments.
  if (isa<FunctionAttr>(objectAttr))
    slotsInStructIndex--;

  DictionaryAttr initMap = constObjectAttrInterface.getSlots();
  for (auto [index, slotName] : llvm::enumerate(
           cast<TypeAttrInterface>(typeObject).getInstanceSlots())) {
    Value value;
    Attribute element = initMap.get(cast<Py::StrAttr>(slotName).getValue());
    if (!element)
      value = builder.create<LLVM::ZeroOp>(global.getLoc(), m_objectPtrType);
    else
      value = getConstant(global.getLoc(), builder, element);

    undef = builder.create<LLVM::InsertValueOp>(
        global.getLoc(), undef, value,
        ArrayRef<std::int64_t>{slotsInStructIndex,
                               static_cast<std::int64_t>(index)});
  }

  builder.create<LLVM::ReturnOp>(global.getLoc(), undef);
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder,
                               Py::StrAttr attr, Value undef, LLVM::GlobalOp) {
  StringRef values = attr.getValue();
  auto sizeConstant = builder.create<LLVM::ConstantOp>(
      loc, m_typeConverter.getIndexType(),
      builder.getI64IntegerAttr(values.size()));
  undef = builder.create<LLVM::InsertValueOp>(
      loc, undef, sizeConstant, llvm::ArrayRef<std::int64_t>{1, 0});
  undef = builder.create<LLVM::InsertValueOp>(
      loc, undef, sizeConstant, llvm::ArrayRef<std::int64_t>{1, 1});

  IntegerType elementType = builder.getI8Type();

  StringAttr strAttr = builder.getStringAttr(values);
  LLVM::GlobalOp bufferObject = m_globalBuffers.lookup(strAttr);
  if (!bufferObject) {
    OpBuilder::InsertionGuard bufferGuard{builder};
    builder.setInsertionPointToStart(
        cast<ModuleOp>(m_symbolTable.getOp()).getBody());

    bufferObject = builder.create<LLVM::GlobalOp>(
        loc, LLVM::LLVMArrayType::get(elementType, values.size()),
        /*isConstant=*/true, LLVM::Linkage::Private, "buffer$", strAttr,
        /*alignment=*/0, /*addrSpace=*/0, /*dsoLocal=*/true);
    bufferObject.setUnnamedAddrAttr(LLVM::UnnamedAddrAttr::get(
        builder.getContext(), LLVM::UnnamedAddr::Global));

    m_symbolTable.insert(bufferObject);
    m_globalBuffers.insert({strAttr, bufferObject});
  }
  auto bufferAddress = builder.create<LLVM::AddressOfOp>(
      loc, builder.getType<LLVM::LLVMPointerType>(),
      FlatSymbolRefAttr::get(bufferObject));
  return builder.create<LLVM::InsertValueOp>(
      loc, undef, bufferAddress, llvm::ArrayRef<std::int64_t>{1, 2});
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder, TupleAttr attr,
                               Value undef, LLVM::GlobalOp) {
  auto sizeConstant =
      builder.create<LLVM::ConstantOp>(loc, m_typeConverter.getIndexType(),
                                       builder.getI64IntegerAttr(attr.size()));
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, sizeConstant, 1);
  for (auto&& [index, value] : llvm::enumerate(attr)) {
    Value constant = getConstant(loc, builder, value);
    undef = builder.create<LLVM::InsertValueOp>(
        loc, undef, constant,
        llvm::ArrayRef<std::int64_t>{2, static_cast<std::int32_t>(index)});
  }
  return undef;
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder, ListAttr attr,
                               Value undef, LLVM::GlobalOp) {
  auto sizeConstant = builder.create<LLVM::ConstantOp>(
      loc, m_typeConverter.getIndexType(),
      builder.getI64IntegerAttr(attr.getElements().size()));
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, sizeConstant, 1);
  LLVM::GlobalOp tupleObject = m_globalBuffers.lookup(attr);
  if (!tupleObject) {
    OpBuilder::InsertionGuard bufferGuard{builder};
    builder.setInsertionPointToStart(
        cast<ModuleOp>(m_symbolTable.getOp()).getBody());

    tupleObject = builder.create<LLVM::GlobalOp>(
        loc, m_typeConverter.getPyTupleType(attr.getElements().size()), true,
        LLVM::Linkage::Private, "tuple$", /*initializer=*/nullptr,
        /*alignment=*/0, REF_ADDRESS_SPACE, /*dsoLocal=*/true);

    initializeGlobal(tupleObject, builder,
                     Py::TupleAttr::get(attr.getContext(), attr.getElements()));
    tupleObject.setUnnamedAddrAttr(LLVM::UnnamedAddrAttr::get(
        builder.getContext(), LLVM::UnnamedAddr::Global));
    m_symbolTable.insert(tupleObject);
    m_globalBuffers.insert({attr, tupleObject});
  }

  auto address = builder.create<LLVM::AddressOfOp>(
      loc, m_objectPtrType, FlatSymbolRefAttr::get(tupleObject));
  return builder.create<LLVM::InsertValueOp>(loc, undef, address, 2);
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder,
                               Py::FloatAttr attr, Value undef,
                               LLVM::GlobalOp) {
  auto constant = builder.create<LLVM::ConstantOp>(
      loc, builder.getF64Type(),
      builder.getF64FloatAttr(attr.getDoubleValue()));
  return builder.create<LLVM::InsertValueOp>(loc, undef, constant, 1);
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder,
                               IntAttrInterface attr, Value undef,
                               LLVM::GlobalOp global) {
  LLVM::GlobalOp result = m_globalBuffers.lookup(attr);
  if (!result) {
    OpBuilder::InsertionGuard bufferGuard{builder};
    builder.setInsertionPointToStart(
        cast<ModuleOp>(m_symbolTable.getOp()).getBody());

    BigInt bigInt = attr.getInteger();
    unsigned targetSizeTBytes = m_typeConverter.getPlatformABI()
                                    .getSizeT(builder.getContext())
                                    .getIntOrFloatBitWidth() /
                                8;
    std::size_t size = mp_pack_count(&bigInt.getHandle(), /*nails=*/0,
                                     /*size=*/targetSizeTBytes);
    llvm::SmallVector<std::size_t> data(size);
    (void)mp_pack(data.data(), /*maxcount=*/data.size(), /*written=*/nullptr,
                  mp_order::MP_LSB_FIRST, targetSizeTBytes, MP_BIG_ENDIAN,
                  /*nails=*/0, &bigInt.getHandle());
    Type elementType =
        m_typeConverter.getPlatformABI().getSizeT(builder.getContext());
    result = builder.create<LLVM::GlobalOp>(
        loc, LLVM::LLVMArrayType::get(elementType, size), true,
        LLVM::Linkage::Private, "buffer$", /*initializer=*/Attribute{},
        /*alignment=*/0, /*addrSpace=*/0, /*dsoLocal=*/true);
    result.setUnnamedAddrAttr(LLVM::UnnamedAddrAttr::get(
        builder.getContext(), LLVM::UnnamedAddr::Global));
    m_symbolTable.insert(result);
    m_globalBuffers.insert({attr, result});
    builder.setInsertionPointToStart(
        &result.getInitializerRegion().emplaceBlock());
    Value arrayUndef = builder.create<LLVM::UndefOp>(loc, result.getType());
    for (auto&& [index, value] : llvm::enumerate(data)) {
      auto constant = builder.create<LLVM::ConstantOp>(
          loc, elementType, builder.getIntegerAttr(elementType, value));
      arrayUndef =
          builder.create<LLVM::InsertValueOp>(loc, arrayUndef, constant, index);
    }
    builder.create<LLVM::ReturnOp>(loc, arrayUndef);
  }
  unsigned numElements =
      cast<LLVM::LLVMArrayType>(result.getType()).getNumElements();
  appendToGlobalInit(builder, [&] {
    Value mpIntPtr;
    {
      auto toInit = builder.create<LLVM::AddressOfOp>(
          loc, m_objectPtrType, FlatSymbolRefAttr::get(global));
      mpIntPtr = builder.create<LLVM::GEPOp>(
          loc, toInit.getType(), global.getType(), toInit,
          llvm::ArrayRef<LLVM::GEPArg>{0, 1});
    }

    createRuntimeCall(loc, builder, Runtime::mp_init, {mpIntPtr});
    auto count = builder.create<LLVM::ConstantOp>(
        loc, m_typeConverter.getIndexType(), builder.getIndexAttr(numElements));
    Type intType =
        m_typeConverter.getPlatformABI().getInt(builder.getContext());
    auto order = builder.create<LLVM::ConstantOp>(
        loc, intType, builder.getIntegerAttr(intType, mp_order::MP_LSB_FIRST));
    auto size = builder.create<LLVM::ConstantOp>(
        loc, m_typeConverter.getIndexType(),
        builder.getIndexAttr(m_typeConverter.getIndexTypeBitwidth() / 8));
    auto endian = builder.create<LLVM::ConstantOp>(
        loc, intType,
        builder.getIntegerAttr(intType, mp_endian::MP_BIG_ENDIAN));
    auto zero = builder.create<LLVM::ConstantOp>(
        loc, m_typeConverter.getIndexType(), builder.getIndexAttr(0));
    auto buffer = builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()),
        FlatSymbolRefAttr::get(result));
    createRuntimeCall(loc, builder, Runtime::mp_unpack,
                      {mpIntPtr, count, order, size, endian, zero, buffer});
  });
  return undef;
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder, DictAttr attr,
                               Value undef, LLVM::GlobalOp global) {
  auto zeroI = builder.create<LLVM::ConstantOp>(
      loc, m_typeConverter.getIndexType(), builder.getIndexAttr(0));
  auto null = builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
  undef = builder.create<LLVM::InsertValueOp>(
      loc, undef, zeroI, llvm::ArrayRef<std::int64_t>{1, 0});
  undef = builder.create<LLVM::InsertValueOp>(
      loc, undef, zeroI, llvm::ArrayRef<std::int64_t>{1, 1});
  undef = builder.create<LLVM::InsertValueOp>(
      loc, undef, null, llvm::ArrayRef<std::int64_t>{1, 2});
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, zeroI, 2);
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, null, 3);
  if (attr.getKeyValuePairs().empty())
    return undef;

  appendToGlobalInit(builder, [&] {
    auto dictionary = builder.create<LLVM::AddressOfOp>(
        loc, m_objectPtrType, FlatSymbolRefAttr::get(global));
    for (const auto& [key, value] : attr.getKeyValuePairs()) {
      Value keyValue = getConstant(loc, builder, key);
      std::optional<Mem::LayoutType> layoutType = m_typeConverter.getLayoutType(
          cast<ObjectAttrInterface>(key).getTypeObject());
      Value hash;
      if (layoutType == Mem::LayoutType::String) {
        hash = createRuntimeCall(loc, builder, Runtime::pylir_str_hash,
                                 {keyValue});
      } else if (layoutType == Mem::LayoutType::Object) {
        hash = builder.create<LLVM::PtrToIntOp>(
            loc, m_typeConverter.getIndexType(), keyValue);
      } else {
        // TODO: Add more inline hash functions implementations.
        PYLIR_UNREACHABLE;
      }

      Value valueValue = getConstant(loc, builder, value);
      createRuntimeCall(loc, builder, Runtime::pylir_dict_insert_unique,
                        {dictionary, keyValue, hash, valueValue});
    }
  });

  return undef;
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder,
                               Py::TypeAttr attr, Value undef, LLVM::GlobalOp) {
  std::optional<Mem::LayoutType> layoutType =
      m_typeConverter.getLayoutType(attr);
  PYLIR_ASSERT(layoutType);
  {
    Type instanceType = m_typeConverter.mapLayoutTypeToLLVM(*layoutType);
    auto asCount = builder.create<LLVM::ConstantOp>(
        loc, m_typeConverter.getIndexType(),
        builder.getI32IntegerAttr(
            m_typeConverter.getPlatformABI().getSizeOf(instanceType) /
            (m_typeConverter.getPointerBitwidth() / 8)));
    undef = builder.create<LLVM::InsertValueOp>(loc, undef, asCount, 1);
  }

  Value layoutRef =
      getConstant(loc, builder,
                  Mem::layoutTypeToTypeObject(attr.getContext(), *layoutType));
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, layoutRef, 2);
  Value mroConstant = getConstant(loc, builder, attr.getMroTuple());
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, mroConstant, 3);
  Value instanceSlots = getConstant(loc, builder, attr.getInstanceSlots());
  return builder.create<LLVM::InsertValueOp>(loc, undef, instanceSlots, 4);
}

Value CodeGenState::initialize(Location loc, OpBuilder& builder,
                               FunctionAttr attr, Value undef, LLVM::GlobalOp) {
  auto address = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), attr.getValue());
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, address, 1);

  // 'FunctionAttr' does not yet support closure arguments making it therefore
  // 0.
  auto closureSize =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  undef = builder.create<LLVM::InsertValueOp>(loc, undef, closureSize, 2);
  return undef;
}

Value CodeGenState::getConstant(Location loc, OpBuilder& builder,
                                Attribute attribute) {
  if (auto value = dyn_cast<Py::GlobalValueAttr>(attribute))
    return builder.create<LLVM::AddressOfOp>(
        loc, m_objectPtrType,
        FlatSymbolRefAttr::get(getGlobalValue(builder, value)));

  if (isa<Py::UnboundAttr>(attribute))
    return builder.create<LLVM::ZeroOp>(loc, m_objectPtrType);

  return builder.create<LLVM::AddressOfOp>(
      loc, m_objectPtrType,
      FlatSymbolRefAttr::get(getConstantObject(
          builder, cast<ConcreteObjectAttribute>(attribute))));
}

LLVM::GlobalOp
CodeGenState::getConstantObject(mlir::OpBuilder& builder,
                                Py::ConcreteObjectAttribute objectAttr) {
  if (auto globalOp = m_constantObjects.lookup(objectAttr))
    return globalOp;

  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(
      cast<ModuleOp>(m_symbolTable.getOp()).getBody());
  auto type = m_typeConverter.typeOf(objectAttr);
  auto globalOp = builder.create<LLVM::GlobalOp>(
      builder.getUnknownLoc(), type, !needToBeRuntimeInit(objectAttr),
      LLVM::Linkage::Private, "const$", Attribute{}, 0, REF_ADDRESS_SPACE,
      true);
  globalOp.setUnnamedAddrAttr(LLVM::UnnamedAddrAttr::get(
      builder.getContext(), LLVM::UnnamedAddr::Global));
  globalOp.setSectionAttr(globalOp.getConstant()
                              ? m_typeConverter.getConstantSection()
                              : m_typeConverter.getCollectionSection());
  m_symbolTable.insert(globalOp);
  m_constantObjects.insert({objectAttr, globalOp});
  initializeGlobal(globalOp, builder, objectAttr);
  return globalOp;
}

LLVM::GlobalOp
CodeGenState::getGlobalValue(OpBuilder& builder,
                             Py::GlobalValueAttr globalValueAttr) {
  auto [iter, inserted] = m_globalValues.insert({globalValueAttr, nullptr});
  if (!inserted)
    return iter->second;

  Type type;
  if (!globalValueAttr.getInitializer()) {
    // If we don't have an initializer, we use the generic PyObject layout for
    // convenience. Semantically, the type given here has no meaning and LLVM
    // does not care about the type of a declaration.
    type = m_typeConverter.getPyObjectType();
  } else {
    type = m_typeConverter.typeOf(globalValueAttr.getInitializer());
  }

  // Check if the global value appeared in an `py.external`.
  // Gives the symbol name that must be used or a null attribute if there was no
  // such occurrence.
  StringAttr exportedName = m_externalGlobalValues.lookup(globalValueAttr);

  bool constant = globalValueAttr.getConstant();
  if (globalValueAttr.getInitializer()) {
    // If the initializer lowering requires runtime initialization we cannot
    // mark it as constant in the LLVM sense.
    constant =
        constant && !needToBeRuntimeInit(globalValueAttr.getInitializer());
  }
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToEnd(
      cast<ModuleOp>(m_symbolTable.getOp()).getBody());
  auto globalOp = builder.create<LLVM::GlobalOp>(
      builder.getUnknownLoc(), type, constant,
      /*linkage=*/
      exportedName ? LLVM::linkage::Linkage::External
                   : LLVM::linkage::Linkage::Internal,
      /*name=*/exportedName ? exportedName : globalValueAttr.getName(),
      /*value=*/Attribute{}, /*alignment=*/0, REF_ADDRESS_SPACE,
      /*dsoLocal=*/true);
  if (!exportedName) {
    // If not meant to be exported, do an explicit insert into the symbol table.
    // This ensures that the global has a unique name within the symbol table.
    m_symbolTable.insert(globalOp);
  }
  // Set the section according to whether this is const.
  // TODO: Evaluate whether this should be dependent on the const in the LLVM or
  // the Pylir sense.
  if (!globalOp.getConstant())
    globalOp.setSectionAttr(m_typeConverter.getCollectionSection());
  else
    globalOp.setSectionAttr(m_typeConverter.getConstantSection());

  iter->second = globalOp;

  // Fill the initializer region.
  if (globalValueAttr.getInitializer())
    initializeGlobal(globalOp, builder, globalValueAttr.getInitializer());

  return globalOp;
}
