//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirTypeConverter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/TargetParser/Triple.h>

#include <pylir/Optimizer/PylirMem/IR/Value.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>

#include "WinX64.hpp"
#include "X86_64.hpp"

using namespace mlir;
using namespace mlir::LLVM;
using namespace pylir;
using namespace pylir::Py;

pylir::PylirTypeConverter::PylirTypeConverter(mlir::MLIRContext* context,
                                              const llvm::Triple& triple,
                                              llvm::DataLayout dataLayout,
                                              mlir::DataLayout&& mlirDataLayout)
    : mlir::LLVMTypeConverter(
          context,
          [&] {
            mlir::LowerToLLVMOptions options(context);
            options.allocLowering =
                mlir::LowerToLLVMOptions::AllocLowering::None;
            options.dataLayout = dataLayout;
            return options;
          }()),
      m_objectPtrType(
          mlir::LLVM::LLVMPointerType::get(&getContext(), REF_ADDRESS_SPACE)) {
  switch (triple.getArch()) {
  case llvm::Triple::x86_64:
    if (triple.isOSWindows())
      m_cabi = std::make_unique<pylir::WinX64>(std::move(mlirDataLayout));
    else
      m_cabi = std::make_unique<pylir::X86_64>(std::move(mlirDataLayout));

    break;
  default: llvm::errs() << triple.str() << " not yet implemented"; std::abort();
  }

  llvm::StringRef constSectionPrefix;
  llvm::StringRef dataSectionPrefix;
  // MachO requires a segment prefix in front of sections to denote their
  // permissions. constants without relocations go into __TEXT, which is
  // read-only, while __DATA, has read-write permission. See
  // https://developer.apple.com/library/archive/documentation/Performance/Conceptual/CodeFootprint/Articles/MachOOverview.html
  if (triple.isOSBinFormatMachO()) {
    // Pretty much all our constants contain relocations. We therefore put them
    // into __DATA.
    constSectionPrefix = "__DATA,";
    dataSectionPrefix = "__DATA,";
  }

  m_rootSection = mlir::StringAttr::get(context, dataSectionPrefix + "py_root");
  m_collectionSection =
      mlir::StringAttr::get(context, dataSectionPrefix + "py_coll");
  m_constantSection =
      mlir::StringAttr::get(context, constSectionPrefix + "py_const");

  addConversion([&](pylir::Py::DynamicType) {
    return mlir::LLVM::LLVMPointerType::get(&getContext(), REF_ADDRESS_SPACE);
  });
  addConversion([&](pylir::Mem::MemoryType) {
    return mlir::LLVM::LLVMPointerType::get(&getContext(), REF_ADDRESS_SPACE);
  });
}

namespace {
std::string slotSizeNameSuffix(std::optional<unsigned int> slotSize) {
  if (!slotSize)
    return {};

  return "[" + std::to_string(*slotSize) + "]";
}

mlir::LLVM::LLVMArrayType getSlotEpilogue(mlir::MLIRContext* context,
                                          unsigned slotSize = 0) {
  return mlir::LLVM::LLVMArrayType::get(
      mlir::LLVM::LLVMPointerType::get(context, pylir::REF_ADDRESS_SPACE),
      slotSize);
}

mlir::LLVM::LLVMStructType
lazyInitStructType(mlir::MLIRContext* context, llvm::StringRef name,
                   std::optional<unsigned int> slotSize,
                   llvm::ArrayRef<mlir::Type> body) {
  llvm::SmallString<16> storage;
  auto type = mlir::LLVM::LLVMStructType::getIdentified(
      context, (name + slotSizeNameSuffix(slotSize)).toStringRef(storage));
  if (type.isInitialized())
    return type;

  auto temp = llvm::to_vector(body);
  temp.push_back(getSlotEpilogue(context, slotSize.value_or(0)));
  [[maybe_unused]] mlir::LogicalResult result = type.setBody(temp, false);
  PYLIR_ASSERT(mlir::succeeded(result));
  return type;
}
} // namespace

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getPyObjectType(
    std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyObject", slotSize,
                            {m_objectPtrType});
}

unsigned
pylir::PylirTypeConverter::getClosureArgsBytes(TypeRange closureArgsTypes) {
  // Note that this is the same calculation as for the size of a struct type
  // except that we do not care about padding at the end of the struct.
  unsigned byteCount = 0;
  for (Type type : closureArgsTypes) {
    byteCount = llvm::alignTo(byteCount, getPlatformABI().getAlignOf(type));
    byteCount += getPlatformABI().getSizeOf(type);
  }
  return byteCount;
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getPyFunctionType(
    std::optional<unsigned int> slotSize, TypeRange closureArgsTypes) {
  assert((slotSize || closureArgsTypes.empty()) &&
         "slot size must be present if specifying closure arguments");

  std::string name = "PyFunction";

  llvm::raw_string_ostream ss(name);
  if (!closureArgsTypes.empty()) {
    ss << '[';
    llvm::interleaveComma(closureArgsTypes, ss);
    ss << ']';
  }
  ss << slotSizeNameSuffix(slotSize);
  auto type = LLVMStructType::getIdentified(&getContext(), name);
  if (type.isInitialized())
    return type;

  SmallVector<Type> body{
      m_objectPtrType,
      mlir::LLVM::LLVMPointerType::get(&getContext()),
      IntegerType::get(&getContext(), 32),
  };
  body.push_back(getSlotEpilogue(&getContext(), slotSize.value_or(0)));
  llvm::append_range(body, closureArgsTypes);
  body.push_back(LLVM::LLVMArrayType::get(
      IntegerType::get(&getContext(), 8),
      llvm::divideCeil(getClosureArgsBytes(closureArgsTypes),
                       8 * m_cabi->getSizeOf(m_objectPtrType))));

  [[maybe_unused]] LogicalResult result =
      type.setBody(body, /*isPacked=*/false);
  PYLIR_ASSERT(succeeded(result));
  return type;
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::getPyTupleType(std::optional<unsigned int> length) {
  return lazyInitStructType(&getContext(), "PyTuple", length,
                            {m_objectPtrType, getIndexType()});
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::getPyListType(std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyList", slotSize,
                            {m_objectPtrType, getIndexType(), m_objectPtrType});
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getBufferComponent() {
  return mlir::LLVM::LLVMStructType::getLiteral(
      &getContext(), {getIndexType(), getIndexType(),
                      mlir::LLVM::LLVMPointerType::get(&getContext())});
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::getPyDictType(std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyDict", slotSize,
                            {m_objectPtrType, getBufferComponent(),
                             getIndexType(),
                             mlir::LLVM::LLVMPointerType::get(&getContext())});
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getPyStringType(
    std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyString", slotSize,
                            {m_objectPtrType, getBufferComponent()});
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getMPInt() {
  auto mpInt =
      mlir::LLVM::LLVMStructType::getIdentified(&getContext(), "mp_int");
  if (!mpInt.isInitialized()) {
    [[maybe_unused]] auto result = mpInt.setBody(
        {m_cabi->getInt(&getContext()), m_cabi->getInt(&getContext()),
         mlir::LLVM::LLVMPointerType::get(&getContext()),
         m_cabi->getInt(&getContext())},
        false);
    PYLIR_ASSERT(mlir::succeeded(result));
  }
  return mpInt;
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::getPyIntType(std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyInt", slotSize,
                            {m_objectPtrType, getMPInt()});
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getPyFloatType(
    std::optional<unsigned int> slotSize) {
  return lazyInitStructType(
      &getContext(), "PyFloat", slotSize,
      {m_objectPtrType, mlir::Float64Type::get(&getContext())});
}

mlir::LLVM::LLVMStructType pylir::PylirTypeConverter::getPyBaseExceptionType(
    std::optional<unsigned int> slotSize) {
  return lazyInitStructType(
      &getContext(), "PyBaseException", slotSize,
      {m_objectPtrType,
       mlir::IntegerType::get(&getContext(), getPointerBitwidth()),
       getUnwindHeaderType(), mlir::IntegerType::get(&getContext(), 32)});
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::getPyTypeType(std::optional<unsigned int> slotSize) {
  return lazyInitStructType(&getContext(), "PyType", slotSize,
                            {m_objectPtrType, getIndexType(), m_objectPtrType,
                             m_objectPtrType, m_objectPtrType});
}

std::optional<pylir::Mem::LayoutType>
pylir::PylirTypeConverter::getLayoutType(mlir::Attribute attr) {
  return pylir::Mem::getLayoutType(attr, &m_layoutTypeCache);
}

mlir::LLVM::LLVMStructType
pylir::PylirTypeConverter::typeOf(pylir::Py::ObjectAttrInterface objectAttr) {
  unsigned count = cast<TypeAttrInterface>(objectAttr.getTypeObject())
                       .getInstanceSlots()
                       .size();
  return llvm::TypeSwitch<pylir::Py::ObjectAttrInterface,
                          mlir::LLVM::LLVMStructType>(objectAttr)
      .Case([&](pylir::Py::TupleAttr attr) {
        PYLIR_ASSERT(count == 0);
        return getPyTupleType(attr.size());
      })
      .Case([&](pylir::Py::ListAttr) { return getPyListType(count); })
      .Case([&](pylir::Py::StrAttr) { return getPyStringType(count); })
      .Case([&](pylir::Py::TypeAttr) { return getPyTypeType(count); })
      .Case([&](pylir::Py::FunctionAttr) { return getPyFunctionType(count); })
      .Case([&](pylir::Py::IntAttr) { return getPyIntType(count); })
      .Case([&](pylir::Py::BoolAttr) { return getPyIntType(count); })
      .Case([&](pylir::Py::FloatAttr) { return getPyFloatType(count); })
      .Case([&](pylir::Py::DictAttr) { return getPyDictType(count); })
      .Default([&](auto) { return getPyObjectType(count); });
}
