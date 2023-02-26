//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGenState.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirMem/IR/Value.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>

void pylir::CodeGenState::appendToGlobalInit(mlir::OpBuilder& builder, llvm::function_ref<void()> section)
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

mlir::ArrayAttr pylir::CodeGenState::getTBAAAccess(TbaaAccessType accessType)
{
    if (accessType == TbaaAccessType::None)
    {
        return mlir::ArrayAttr::get(&m_typeConverter.getContext(), {});
    }

    mlir::OpBuilder builder(&m_typeConverter.getContext());
    if (!m_tbaaSymbolTable)
    {
        m_tbaaRegion = builder.create<mlir::LLVM::MetadataOp>(builder.getUnknownLoc(), "tbaa");
        m_tbaaSymbolTable.emplace(*m_tbaaRegion);
        m_tbaaRoot = builder.create<mlir::LLVM::TBAARootMetadataOp>(builder.getUnknownLoc(), "root", "Pylir TBAA Root");
        m_tbaaSymbolTable->insert(m_tbaaRoot);
    }
    std::string tbaaAccessTypeString;
    switch (accessType)
    {
        case TbaaAccessType::None: PYLIR_UNREACHABLE;
        case TbaaAccessType::Slots: tbaaAccessTypeString = "Python Object Slots"; break;
        case TbaaAccessType::TypeObject: tbaaAccessTypeString = "Python Type Object"; break;
        case TbaaAccessType::TupleElements: tbaaAccessTypeString = "Python Tuple Elements"; break;
        case TbaaAccessType::ListTupleMember: tbaaAccessTypeString = "Python List Tuple"; break;
        case TbaaAccessType::TypeMroMember: tbaaAccessTypeString = "Python Type MRO"; break;
        case TbaaAccessType::TypeSlotsMember: tbaaAccessTypeString = "Python Type Instance Slots"; break;
        case TbaaAccessType::TypeOffset: tbaaAccessTypeString = "Python Type Offset"; break;
        case TbaaAccessType::Handle: tbaaAccessTypeString = "Python Handle"; break;
        case TbaaAccessType::TupleSize: tbaaAccessTypeString = "Python Tuple Size"; break;
        case TbaaAccessType::ListSize: tbaaAccessTypeString = "Python List Size"; break;
        case TbaaAccessType::DictSize: tbaaAccessTypeString = "Python Dict Size"; break;
        case TbaaAccessType::StringSize: tbaaAccessTypeString = "Python String Size"; break;
        case TbaaAccessType::StringCapacity: tbaaAccessTypeString = "Python String Capacity"; break;
        case TbaaAccessType::StringElementPtr: tbaaAccessTypeString = "Python String Element Ptr"; break;
        case TbaaAccessType::FloatValue: tbaaAccessTypeString = "Python Float Value"; break;
        case TbaaAccessType::FunctionPointer: tbaaAccessTypeString = "Python Function Pointer"; break;
    }

    std::string accessName = tbaaAccessTypeString + " access";
    std::string typeName = tbaaAccessTypeString + " type";
    if (auto existing = m_tbaaSymbolTable->lookup<mlir::LLVM::TBAATagOp>(accessName))
    {
        return builder.getArrayAttr(mlir::SymbolRefAttr::get(m_tbaaRegion->getSymNameAttr(),
                                                             mlir::FlatSymbolRefAttr::get(existing.getSymNameAttr())));
    }

    auto type = builder.create<mlir::LLVM::TBAATypeDescriptorOp>(
        builder.getUnknownLoc(), /*sym_name=*/typeName, /*identity=*/builder.getStringAttr(typeName),
        builder.getArrayAttr({mlir::FlatSymbolRefAttr::get(m_tbaaRoot)}), builder.getDenseI64ArrayAttr({0}));
    m_tbaaSymbolTable->insert(type);

    auto tag = builder.create<mlir::LLVM::TBAATagOp>(builder.getUnknownLoc(), /*sym_name=*/accessName,
                                                     /*base_type=*/typeName, /*access_type=*/typeName, 0);
    m_tbaaSymbolTable->insert(tag);
    return builder.getArrayAttr(
        mlir::SymbolRefAttr::get(m_tbaaRegion->getSymNameAttr(), mlir::FlatSymbolRefAttr::get(tag.getSymNameAttr())));
}

mlir::Value pylir::CodeGenState::createRuntimeCall(mlir::Location loc, mlir::OpBuilder& builder,
                                                   CodeGenState::Runtime func, mlir::ValueRange args)
{
    PlatformABI& abi = m_typeConverter.getPlatformABI();
    mlir::MLIRContext* context = builder.getContext();
    auto pointerType = builder.getType<mlir::LLVM::LLVMPointerType>();

    mlir::Type returnType;
    mlir::NamedAttrList resultAttrs;
    llvm::SmallVector<mlir::Type> argumentTypes;
    std::string functionName;
    llvm::SmallVector<llvm::StringRef> passThroughAttributes;
    switch (func)
    {
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
            resultAttrs.append(mlir::LLVM::LLVMDialect::getNoAliasAttrName(), mlir::UnitAttr::get(context));
            break;
        case Runtime::mp_init_u64:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, builder.getI64Type()};
            functionName = "mp_init_u64";
            passThroughAttributes = {"gc-leaf-function", "nounwind"};
            break;
        case Runtime::mp_init_i64:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
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
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType};
            functionName = "pylir_print";
            passThroughAttributes = {"gc-leaf-function", "nounwind"};
            break;
        case Runtime::pylir_raise:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType};
            functionName = "pylir_raise";
            passThroughAttributes = {"noreturn"};
            break;
        case Runtime::mp_init:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType};
            functionName = "mp_init";
            passThroughAttributes = {"gc-leaf-function", "nounwind"};
            break;
        case Runtime::mp_unpack:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType,     m_typeConverter.getIndexType(),
                             abi.getInt(context), m_typeConverter.getIndexType(),
                             abi.getInt(context), m_typeConverter.getIndexType(),
                             pointerType};
            functionName = "mp_unpack";
            passThroughAttributes = {"gc-leaf-function", "nounwind"};
            break;
        case Runtime::mp_radix_size_overestimate:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, abi.getInt(context), m_objectPtrType};
            functionName = "mp_radix_size_overestimate";
            passThroughAttributes = {"gc-leaf-function", "nounwind"};
            break;
        case Runtime::mp_to_radix:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, pointerType, m_typeConverter.getIndexType(), m_objectPtrType,
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
            returnType = mlir::LLVM::LLVMVoidType::get(context);
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
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context)};
            functionName = "pylir_dict_erase";
            break;
        case Runtime::pylir_dict_insert:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context), m_objectPtrType};
            functionName = "pylir_dict_insert";
            break;
        case Runtime::pylir_dict_insert_unique:
            returnType = mlir::LLVM::LLVMVoidType::get(context);
            argumentTypes = {m_objectPtrType, m_objectPtrType, abi.getSizeT(context), m_objectPtrType};
            functionName = "pylir_dict_insert_unique";
            break;
    }

    auto module = mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp());
    auto llvmFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName);
    if (!llvmFunc)
    {
        mlir::OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPointToEnd(module.getBody());
        llvmFunc = abi.declareFunc(builder, loc, returnType, functionName, argumentTypes);
        if (!passThroughAttributes.empty())
        {
            llvmFunc.setPassthroughAttr(builder.getStrArrayAttr(passThroughAttributes));
        }
        if (!resultAttrs.empty())
        {
            llvmFunc.setResultAttrs(0, resultAttrs);
        }
    }
    return abi.callFunc(builder, loc, llvmFunc, args);
}

void pylir::CodeGenState::initializeGlobal(mlir::LLVM::GlobalOp global, mlir::OpBuilder& builder,
                                           Py::ObjectAttrInterface objectAttr)
{
    builder.setInsertionPointToStart(&global.getInitializerRegion().emplaceBlock());
    mlir::Value undef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), global.getType());
    auto typeObject = objectAttr.getTypeObject();
    PYLIR_ASSERT(typeObject);
    PYLIR_ASSERT(typeObject.getSymbol().getInitializerAttr() && "Type objects can't be a declaration");

    auto typeObj =
        builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), m_objectPtrType, objectAttr.getTypeObject().getRef());
    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, typeObj, 0);
    llvm::TypeSwitch<Py::ObjectAttrInterface>(objectAttr)
        .Case(
            [&](Py::StrAttr attr)
            {
                auto values = attr.getValue();
                auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                    global.getLoc(), m_typeConverter.getIndexType(), builder.getI64IntegerAttr(values.size()));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                  llvm::ArrayRef<std::int64_t>{1, 0});
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant,
                                                                  llvm::ArrayRef<std::int64_t>{1, 1});

                auto elementType = builder.getI8Type();

                auto strAttr = builder.getStringAttr(values);
                auto bufferObject = m_globalBuffers.lookup(strAttr);
                if (!bufferObject)
                {
                    mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                    builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                    bufferObject = builder.create<mlir::LLVM::GlobalOp>(
                        global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, values.size()), true,
                        mlir::LLVM::Linkage::Private, "buffer$", strAttr, 0, 0, true);
                    bufferObject.setUnnamedAddrAttr(
                        mlir::LLVM::UnnamedAddrAttr::get(builder.getContext(), mlir::LLVM::UnnamedAddr::Global));
                    m_symbolTable.insert(bufferObject);
                    m_globalBuffers.insert({strAttr, bufferObject});
                }
                auto bufferAddress = builder.create<mlir::LLVM::AddressOfOp>(
                    global.getLoc(), builder.getType<mlir::LLVM::LLVMPointerType>(),
                    mlir::FlatSymbolRefAttr::get(bufferObject));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, bufferAddress,
                                                                  llvm::ArrayRef<std::int64_t>{1, 2});
            })
        .Case(
            [&](Py::TupleAttr attr)
            {
                auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                    global.getLoc(), m_typeConverter.getIndexType(), builder.getI64IntegerAttr(attr.size()));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant, 1);
                for (const auto& iter : llvm::enumerate(attr))
                {
                    auto constant = getConstant(global.getLoc(), builder, iter.value());
                    undef = builder.create<mlir::LLVM::InsertValueOp>(
                        global.getLoc(), undef, constant,
                        llvm::ArrayRef<std::int64_t>{2, static_cast<std::int32_t>(iter.index())});
                }
            })
        .Case(
            [&](Py::ListAttr attr)
            {
                auto sizeConstant = builder.create<mlir::LLVM::ConstantOp>(
                    global.getLoc(), m_typeConverter.getIndexType(), builder.getI64IntegerAttr(attr.getValue().size()));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, sizeConstant, 1);
                auto tupleObject = m_globalBuffers.lookup(attr);
                if (!tupleObject)
                {
                    mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                    builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                    tupleObject = builder.create<mlir::LLVM::GlobalOp>(
                        global.getLoc(), m_typeConverter.getPyTupleType(attr.getValue().size()), true,
                        mlir::LLVM::Linkage::Private, "tuple$", nullptr, 0, REF_ADDRESS_SPACE, true);
                    initializeGlobal(tupleObject, builder, Py::TupleAttr::get(attr.getContext(), attr.getValue()));
                    tupleObject.setUnnamedAddrAttr(
                        mlir::LLVM::UnnamedAddrAttr::get(builder.getContext(), mlir::LLVM::UnnamedAddr::Global));
                    m_symbolTable.insert(tupleObject);
                    m_globalBuffers.insert({attr, tupleObject});
                }
                auto address = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), m_objectPtrType,
                                                                       mlir::FlatSymbolRefAttr::get(tupleObject));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, address, 2);
            })
        .Case(
            [&](Py::FloatAttr floatAttr)
            {
                auto constant = builder.create<mlir::LLVM::ConstantOp>(
                    global.getLoc(), builder.getF64Type(), builder.getF64FloatAttr(floatAttr.getDoubleValue()));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, constant, 1);
            })
        .Case(
            [&](Py::IntAttr integer)
            {
                auto result = m_globalBuffers.lookup(integer);
                if (!result)
                {
                    mlir::OpBuilder::InsertionGuard bufferGuard{builder};
                    builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
                    auto bigInt = integer.getValue();
                    auto targetSizeTBytes =
                        m_typeConverter.getPlatformABI().getSizeT(builder.getContext()).getIntOrFloatBitWidth() / 8;
                    auto size = mp_pack_count(&bigInt.getHandle(), 0, targetSizeTBytes);
                    llvm::SmallVector<std::size_t> data(size);
                    (void)mp_pack(data.data(), data.size(), nullptr, mp_order::MP_LSB_FIRST, targetSizeTBytes,
                                  MP_BIG_ENDIAN, 0, &bigInt.getHandle());
                    auto elementType = m_typeConverter.getPlatformABI().getSizeT(builder.getContext());
                    result = builder.create<mlir::LLVM::GlobalOp>(
                        global.getLoc(), mlir::LLVM::LLVMArrayType::get(elementType, size), true,
                        mlir::LLVM::Linkage::Private, "buffer$", mlir::Attribute{}, 0, 0, true);
                    result.setUnnamedAddrAttr(
                        mlir::LLVM::UnnamedAddrAttr::get(builder.getContext(), mlir::LLVM::UnnamedAddr::Global));
                    m_symbolTable.insert(result);
                    m_globalBuffers.insert({integer, result});
                    builder.setInsertionPointToStart(&result.getInitializerRegion().emplaceBlock());
                    mlir::Value arrayUndef = builder.create<mlir::LLVM::UndefOp>(global.getLoc(), result.getType());
                    for (const auto& element : llvm::enumerate(data))
                    {
                        auto constant = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), elementType, builder.getIntegerAttr(elementType, element.value()));
                        arrayUndef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), arrayUndef, constant,
                                                                               element.index());
                    }
                    builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), arrayUndef);
                }
                auto numElements = result.getType().template cast<mlir::LLVM::LLVMArrayType>().getNumElements();
                appendToGlobalInit(
                    builder,
                    [&]
                    {
                        mlir::Value mpIntPtr;
                        {
                            auto toInit = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), m_objectPtrType,
                                                                                  mlir::FlatSymbolRefAttr::get(global));
                            mpIntPtr =
                                builder.create<mlir::LLVM::GEPOp>(global.getLoc(), toInit.getType(), global.getType(),
                                                                  toInit, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
                        }

                        createRuntimeCall(global.getLoc(), builder, Runtime::mp_init, {mpIntPtr});
                        auto count = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), m_typeConverter.getIndexType(), builder.getIndexAttr(numElements));
                        auto intType = m_typeConverter.getPlatformABI().getInt(builder.getContext());
                        auto order = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), intType, builder.getIntegerAttr(intType, mp_order::MP_LSB_FIRST));
                        auto size = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), m_typeConverter.getIndexType(),
                            builder.getIndexAttr(m_typeConverter.getIndexTypeBitwidth() / 8));
                        auto endian = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), intType, builder.getIntegerAttr(intType, mp_endian::MP_BIG_ENDIAN));
                        auto zero = builder.create<mlir::LLVM::ConstantOp>(
                            global.getLoc(), m_typeConverter.getIndexType(), builder.getIndexAttr(0));
                        auto buffer = builder.create<mlir::LLVM::AddressOfOp>(
                            global.getLoc(), mlir::LLVM::LLVMPointerType::get(builder.getContext()),
                            mlir::FlatSymbolRefAttr::get(result));
                        createRuntimeCall(global.getLoc(), builder, Runtime::mp_unpack,
                                          {mpIntPtr, count, order, size, endian, zero, buffer});
                    });
            })
        .Case(
            [&](Py::DictAttr dict)
            {
                auto zeroI = builder.create<mlir::LLVM::ConstantOp>(global.getLoc(), m_typeConverter.getIndexType(),
                                                                    builder.getIndexAttr(0));
                auto null = builder.create<mlir::LLVM::NullOp>(global.getLoc(),
                                                               mlir::LLVM::LLVMPointerType::get(builder.getContext()));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI,
                                                                  llvm::ArrayRef<std::int64_t>{1, 0});
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI,
                                                                  llvm::ArrayRef<std::int64_t>{1, 1});
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, null,
                                                                  llvm::ArrayRef<std::int64_t>{1, 2});
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, zeroI, 2);
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, null, 3);
                if (dict.getKeyValuePairs().empty())
                {
                    return;
                }
                appendToGlobalInit(
                    builder,
                    [&]
                    {
                        auto dictionary = builder.create<mlir::LLVM::AddressOfOp>(global.getLoc(), m_objectPtrType,
                                                                                  mlir::FlatSymbolRefAttr::get(global));
                        for (const auto& [key, value] : dict.getKeyValuePairs())
                        {
                            auto keyValue = getConstant(global.getLoc(), builder, key);
                            auto layoutType = m_typeConverter.getLayoutType(
                                Py::ref_cast<Py::ObjectAttrInterface>(key, false).getTypeObject());
                            mlir::Value hash;
                            if (layoutType == Mem::LayoutType::String)
                            {
                                hash = createRuntimeCall(global.getLoc(), builder, Runtime::pylir_str_hash, {keyValue});
                            }
                            else if (layoutType == Mem::LayoutType::Object)
                            {
                                hash = builder.create<mlir::LLVM::PtrToIntOp>(global.getLoc(),
                                                                              m_typeConverter.getIndexType(), keyValue);
                            }
                            else
                            {
                                // TODO: Add more inline hash functions implementations.
                                PYLIR_UNREACHABLE;
                            }
                            auto valueValue = getConstant(global.getLoc(), builder, value);
                            createRuntimeCall(global.getLoc(), builder, Runtime::pylir_dict_insert_unique,
                                              {dictionary, keyValue, hash, valueValue});
                        }
                    });
            })
        .Case(
            [&](Py::TypeAttr attr)
            {
                auto layoutType = m_typeConverter.getLayoutType(
                    Py::RefAttr::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(global)));
                PYLIR_ASSERT(layoutType);
                {
                    auto instanceType = m_typeConverter.mapLayoutTypeToLLVM(*layoutType);
                    auto asCount = builder.create<mlir::LLVM::ConstantOp>(
                        global.getLoc(), m_typeConverter.getIndexType(),
                        builder.getI32IntegerAttr(m_typeConverter.getPlatformABI().getSizeOf(instanceType)
                                                  / (m_typeConverter.getPointerBitwidth() / 8)));
                    undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, asCount, 1);
                }

                auto layoutRef =
                    getConstant(global.getLoc(), builder, Mem::layoutTypeToTypeObject(attr.getContext(), *layoutType));
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, layoutRef, 2);
                auto mroConstant = getConstant(global.getLoc(), builder, attr.getMroTuple());
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, mroConstant, 3);
                auto instanceSlots = getConstant(global.getLoc(), builder, attr.getInstanceSlots());
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, instanceSlots, 4);
            })
        .Case(
            [&](Py::FunctionAttr function)
            {
                auto address = builder.create<mlir::LLVM::AddressOfOp>(
                    global.getLoc(), mlir::LLVM::LLVMPointerType::get(builder.getContext()), function.getValue());
                undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, address, 1);
            });

    auto initMap = objectAttr.getSlots();
    for (auto [index, slotName] : llvm::enumerate(Py::ref_cast<Py::TypeAttr>(typeObject).getInstanceSlots()))
    {
        mlir::Value value;
        if (auto element = initMap.get(slotName.cast<Py::StrAttr>().getValue()); !element)
        {
            value = builder.create<mlir::LLVM::NullOp>(global.getLoc(), m_objectPtrType);
        }
        else
        {
            value = getConstant(global.getLoc(), builder, element);
        }
        auto indices = {
            static_cast<std::int64_t>(global.getType().cast<mlir::LLVM::LLVMStructType>().getBody().size() - 1),
            static_cast<std::int64_t>(index)};
        undef = builder.create<mlir::LLVM::InsertValueOp>(global.getLoc(), undef, value, indices);
    }

    builder.create<mlir::LLVM::ReturnOp>(global.getLoc(), undef);
}

mlir::Value pylir::CodeGenState::getConstant(mlir::Location loc, mlir::OpBuilder& builder, mlir::Attribute attribute)
{
    mlir::LLVM::AddressOfOp address;
    if (auto ref = attribute.dyn_cast<Py::RefAttr>())
    {
        return builder.create<mlir::LLVM::AddressOfOp>(loc, m_objectPtrType, ref.getRef());
    }
    if (attribute.isa<Py::UnboundAttr>())
    {
        return builder.create<mlir::LLVM::NullOp>(loc, m_objectPtrType);
    }

    return address = builder.create<mlir::LLVM::AddressOfOp>(
               loc, m_objectPtrType,
               mlir::FlatSymbolRefAttr::get(createGlobalConstant(builder, attribute.cast<Py::ObjectAttrInterface>())));
}

mlir::LLVM::GlobalOp pylir::CodeGenState::createGlobalConstant(mlir::OpBuilder& builder,
                                                               Py::ObjectAttrInterface objectAttr)
{
    if (auto globalOp = m_globalConstants.lookup(objectAttr))
    {
        return globalOp;
    }
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(mlir::cast<mlir::ModuleOp>(m_symbolTable.getOp()).getBody());
    auto type = m_typeConverter.typeOf(objectAttr);
    auto globalOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), type,
                                                         !needToBeRuntimeInit(objectAttr), mlir::LLVM::Linkage::Private,
                                                         "const$", mlir::Attribute{}, 0, REF_ADDRESS_SPACE, true);
    globalOp.setUnnamedAddrAttr(
        mlir::LLVM::UnnamedAddrAttr::get(builder.getContext(), mlir::LLVM::UnnamedAddr::Global));
    globalOp.setSectionAttr(globalOp.getConstant() ? m_typeConverter.getConstantSection() :
                                                     m_typeConverter.getCollectionSection());
    m_symbolTable.insert(globalOp);
    m_globalConstants.insert({objectAttr, globalOp});
    initializeGlobal(globalOp, builder, objectAttr);
    return globalOp;
}