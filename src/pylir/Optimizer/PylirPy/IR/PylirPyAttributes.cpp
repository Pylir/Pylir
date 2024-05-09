//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyAttributes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <utility>

#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"
#include "Value.hpp"

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

template <>
struct mlir::FieldParser<llvm::APFloat> {
  static mlir::FailureOr<llvm::APFloat> parse(mlir::AsmParser& parser) {
    double value;
    if (parser.parseFloat(value))
      return mlir::failure();

    return llvm::APFloat(value);
  }
};

template <>
struct mlir::FieldParser<pylir::BigInt> {
  static mlir::FailureOr<pylir::BigInt> parse(mlir::AsmParser& parser) {
    llvm::APInt apInt;
    if (parser.parseInteger(apInt))
      return mlir::failure();

    llvm::SmallString<10> str;
    apInt.toStringSigned(str);
    return pylir::BigInt(std::string{str.data(), str.size()});
  }
};

namespace pylir {
llvm::hash_code hash_value(const pylir::BigInt& bigInt) {
  auto count = mp_sbin_size(&bigInt.getHandle());
  llvm::SmallVector<std::uint8_t, 10> data(count);
  auto result = mp_to_sbin(&bigInt.getHandle(), data.data(), count, nullptr);
  PYLIR_ASSERT(result == MP_OKAY);
  return llvm::hash_value(llvm::ArrayRef(data));
}
} // namespace pylir

bool ConcreteObjectAttribute::classof(mlir::Attribute attribute) {
  return llvm::isa<ObjectAttr, IntAttr, BoolAttr, FloatAttr, StrAttr, TupleAttr,
                   ListAttr, DictAttr, FunctionAttr, TypeAttr>(attribute);
}

//===----------------------------------------------------------------------===//
// IntAttr
//===----------------------------------------------------------------------===//

Attribute IntAttr::getCanonicalAttribute() const {
  return FractionalAttr::get(getContext(), getInteger(), BigInt(1));
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

Attribute Py::BoolAttr::getCanonicalAttribute() const {
  return FractionalAttr::get(getContext(), BigInt(getValue() ? 1 : 0),
                             BigInt(1));
}

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

Attribute Py::FloatAttr::getCanonicalAttribute() const {
  auto [nom, denom] = toRatio(getDoubleValue());
  return FractionalAttr::get(getContext(), std::move(nom), std::move(denom));
}

//===----------------------------------------------------------------------===//
// StrAttr
//===----------------------------------------------------------------------===//

Attribute StrAttr::getCanonicalAttribute() const {
  return StringAttr::get(getContext(), getValue());
}

//===----------------------------------------------------------------------===//
// RefAttr
//===----------------------------------------------------------------------===//

namespace pylir::Py::detail {
struct RefAttrStorage : mlir::AttributeStorage {
  using KeyTy = std::tuple<mlir::FlatSymbolRefAttr>;

  explicit RefAttrStorage(mlir::FlatSymbolRefAttr identity)
      : identity(identity) {}

  bool operator==(const KeyTy& key) const {
    return std::get<0>(key) == identity;
  }

  static RefAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                   const KeyTy& key) {
    return new (allocator.allocate<RefAttrStorage>())
        RefAttrStorage(std::get<0>(key));
  }

  [[nodiscard]] KeyTy getAsKey() const {
    return {mlir::cast<mlir::FlatSymbolRefAttr>(identity)};
  }

  mlir::SymbolRefAttr identity;
  mlir::Operation* value{};
};
} // namespace pylir::Py::detail

namespace {

#define GEN_WRAP_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyWrapInterfaces.h.inc"

template <class Interface>
struct GlobalValueAttrWrapInterface
    : WrapInterface<GlobalValueAttrWrapInterface<Interface>, Interface> {
  /// Returns the underlying instance that all interface methods are forwarded
  /// to by default.
  Interface getUnderlying(Attribute attribute) const {
    return cast<Interface>(cast<GlobalValueAttr>(attribute).getInitializer());
  }

  /// Returns true if the `GlobalValueAttr` implements `Interface`.
  bool canImplement(Attribute thisAttr, std::in_place_type_t<Interface>) const {
    auto globalValueAttr = cast<GlobalValueAttr>(thisAttr);

    // If the interface inherits from `ConstObjectAttrInterface` it has an
    // implicit conversion to it and it is known that the symbol has to be
    // constant for the cast to be valid.
    if constexpr (std::is_convertible_v<Interface, ConstObjectAttrInterface>)
      if (!globalValueAttr.getConstant())
        return false;

    return isa_and_nonnull<Interface>(globalValueAttr.getInitializer());
  }
};

} // namespace

template <class... Interfaces>
static void addGlobalValueAttrInterfaces(MLIRContext* context) {
  GlobalValueAttr::attachInterface<GlobalValueAttrWrapInterface<Interfaces>...>(
      *context);
}

//===----------------------------------------------------------------------===//
// GlobalValueAttr
//===----------------------------------------------------------------------===//

namespace pylir::Py::detail {
struct GlobalValueAttrStorage : mlir::AttributeStorage {
  using KeyTy = std::tuple<llvm::StringRef, bool, ConcreteObjectAttribute>;

  explicit GlobalValueAttrStorage(const KeyTy& key)
      : name(std::get<llvm::StringRef>(key)), constant(std::get<bool>(key)),
        initializer(std::get<ConcreteObjectAttribute>(key)) {}

  bool operator==(const KeyTy& key) const {
    return std::get<llvm::StringRef>(key) == name;
  }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_value(std::get<llvm::StringRef>(key));
  }

  static GlobalValueAttrStorage*
  construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<GlobalValueAttrStorage>())
        GlobalValueAttrStorage(std::make_tuple(
            allocator.copyInto(std::get<llvm::StringRef>(key)),
            std::get<bool>(key), std::get<ConcreteObjectAttribute>(key)));
  }

  [[nodiscard]] KeyTy getAsKey() const {
    return std::make_tuple(name, constant, initializer);
  }

  static KeyTy getKey(llvm::StringRef name) {
    return std::make_tuple(name, false, nullptr);
  }

  mlir::LogicalResult mutate(mlir::AttributeStorageAllocator&, bool c) {
    constant = c;
    return mlir::success();
  }

  mlir::LogicalResult mutate(mlir::AttributeStorageAllocator&,
                             ConcreteObjectAttribute attribute) {
    initializer = attribute;
    return mlir::success();
  }

  llvm::StringRef name;
  bool constant;
  ConcreteObjectAttribute initializer;
};

} // namespace pylir::Py::detail

/// global-value ::= `#py.globalValue` `<` name { (`,` `const`)? (`,`
/// `initializer` `=` attr) }`>`
mlir::Attribute GlobalValueAttr::parse(AsmParser& parser, Type) {
  std::string name;
  if (parser.parseLess() || parser.parseKeywordOrString(&name))
    return nullptr;

  GlobalValueAttr attr = get(parser.getContext(), name);
  FailureOr<AsmParser::CyclicParseReset> reset =
      parser.tryStartCyclicParse(attr);
  if (failed(reset)) {
    if (parser.parseGreater())
      return nullptr;

    return attr;
  }

  // Default values.
  bool constant = false;
  ConcreteObjectAttribute initializer = {};

  while (mlir::succeeded(parser.parseOptionalComma())) {
    llvm::StringRef keyword;
    mlir::ParseResult result =
        mlir::OpAsmParser::KeywordSwitch(parser, &keyword)
            .Case("const",
                  [&](llvm::StringRef, llvm::SMLoc loc) -> mlir::ParseResult {
                    if (constant)
                      return parser.emitError(
                          loc, "'const' cannot be specified more than once");

                    constant = true;
                    return mlir::success();
                  })
            .Case("initializer",
                  [&](llvm::StringRef, llvm::SMLoc loc) -> mlir::ParseResult {
                    if (initializer)
                      return parser.emitError(
                          loc,
                          "'initializer' cannot be specified more than once");

                    return mlir::failure(parser.parseEqual() ||
                                         parser.parseAttribute(initializer));
                  });
    if (result)
      return nullptr;
  }

  if (parser.parseGreater())
    return nullptr;

  attr.setConstant(constant);
  attr.setInitializer(initializer);
  return attr;
}

void GlobalValueAttr::print(AsmPrinter& printer) const {
  printer << '<';

  printer.printKeywordOrString(getName());

  // Break a potential cycle by not printing a nested `#py.globalValue`.
  FailureOr<AsmPrinter::CyclicPrintReset> reset =
      printer.tryStartCyclicPrint(*this);
  if (failed(reset)) {
    printer << '>';
    return;
  }

  if (getConstant())
    printer << ", const";
  if (getInitializer())
    printer << ", initializer = " << getInitializer();

  printer << '>';
}

llvm::StringRef pylir::Py::GlobalValueAttr::getName() const {
  return getImpl()->name;
}

bool pylir::Py::GlobalValueAttr::getConstant() const {
  return getImpl()->constant;
}

void pylir::Py::GlobalValueAttr::setConstant(bool constant) {
  (void)Base::mutate(constant);
}

pylir::Py::ConcreteObjectAttribute
pylir::Py::GlobalValueAttr::getInitializer() const {
  return getImpl()->initializer;
}

void pylir::Py::GlobalValueAttr::setInitializer(
    ConcreteObjectAttribute initializer) {
  (void)Base::mutate(initializer);
}

//===----------------------------------------------------------------------===//
// DictAttr
//===----------------------------------------------------------------------===//

namespace {

std::size_t
lookup(llvm::ArrayRef<std::pair<mlir::Attribute, std::size_t>> normalizedKeys,
       mlir::Attribute key) {
  for (auto bucket = hash_value(key) % normalizedKeys.size();;
       // This is essentially bucket = (bucket + 1) % normalizedKeys.size(), but
       // uses a conditional move instead of a module operation on each
       // iteration.
       bucket = bucket + 1 < normalizedKeys.size() ? bucket + 1 : 0) {
    if (!normalizedKeys[bucket].first || key == normalizedKeys[bucket].first)
      return bucket;
  }
  PYLIR_UNREACHABLE;
}

struct UniqueOutput {
  llvm::SmallVector<std::pair<mlir::Attribute, std::size_t>>
      normalizedKeysUnique;
  llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>>
      keyValuePairsUnique;
};

UniqueOutput unique(llvm::ArrayRef<pylir::Py::DictAttr::Entry> entries) {
  // Doing this step ahead of time with a MapVector purely to get the correct
  // size ahead of time. Changing size would require rehashing.
  llvm::MapVector<mlir::Attribute, std::pair<mlir::Attribute, mlir::Attribute>>
      map;
  for (auto [key, normKey, value] : entries) {
    PYLIR_ASSERT(key && normKey && value);
    auto [iter, inserted] = map.insert({normKey, {key, value}});
    if (!inserted)
      // We have to retain the key, but update the value it is mapped to.
      iter->second.second = value;
  }

  // Our Hashtable has a load factor of 0.9 currently. The tradeoff here is
  // between memory usage and lookup speed, especially in the case of a key not
  // being within the dictionary.
  UniqueOutput result;
  result.normalizedKeysUnique.resize(std::ceil(10 * map.size() / 9.0));
  result.keyValuePairsUnique.reserve(map.size());
  for (auto& [key, pair] : map) {
    result.normalizedKeysUnique[lookup(result.normalizedKeysUnique, key)] = {
        key, result.keyValuePairsUnique.size()};
    result.keyValuePairsUnique.push_back(std::move(pair));
  }
  return result;
}

} // namespace

pylir::Py::DictAttr pylir::Py::DictAttr::get(mlir::MLIRContext* context,
                                             llvm::ArrayRef<Entry> entries) {
  auto result = ::unique(entries);
  return Base::get(context, result.normalizedKeysUnique,
                   result.keyValuePairsUnique);
}

mlir::Attribute pylir::Py::DictAttr::lookup(mlir::Attribute key) const {
  if (getNormalizedKeysInternal().empty())
    return nullptr;

  auto equalsAttrInterface = mlir::dyn_cast<EqualsAttrInterface>(key);
  if (!equalsAttrInterface)
    return nullptr;

  std::size_t bucket = ::lookup(getNormalizedKeysInternal(),
                                equalsAttrInterface.getCanonicalAttribute());
  auto [normalizedKey, index] = getNormalizedKeysInternal()[bucket];
  if (!normalizedKey)
    return nullptr;

  return getKeyValuePairs()[index].second;
}

namespace {

mlir::LogicalResult parseKVPair(
    mlir::AsmParser& parser,
    llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>>& values,
    llvm::SmallVector<std::pair<mlir::Attribute, std::size_t>>&
        normalizedKeys) {
  llvm::SmallVector<pylir::Py::DictAttr::Entry> entries;
  auto parseResult =
      parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces, [&] {
        EqualsAttrInterface key;
        mlir::Attribute value;
        if (parser.parseAttribute(key) || parser.parseKeyword("to") ||
            parser.parseAttribute(value))
          return mlir::failure();

        entries.emplace_back(key, key.getCanonicalAttribute(), value);
        return mlir::success();
      });
  if (mlir::failed(parseResult))
    return mlir::failure();

  auto result = ::unique(entries);
  values = std::move(result.keyValuePairsUnique);
  normalizedKeys = std::move(result.normalizedKeysUnique);

  return mlir::success();
}

void printKVPair(
    mlir::AsmPrinter& printer,
    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> values,
    llvm::ArrayRef<std::pair<mlir::Attribute, std::size_t>>) {
  printer << "{";
  llvm::interleaveComma(values, printer.getStream(),
                        [&](std::pair<mlir::Attribute, mlir::Attribute> pair) {
                          printer << pair.first << " to " << pair.second;
                        });
  printer << "}";
}

} // namespace

//===----------------------------------------------------------------------===//
// FunctionAttr
//===----------------------------------------------------------------------===//

mlir::DictionaryAttr pylir::Py::FunctionAttr::getSlots() const {
  llvm::SmallVector<mlir::NamedAttribute> vector = {
      mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__defaults__"),
                           getDefaults()),
  };
  if (getDict())
    vector.emplace_back(mlir::NamedAttribute(
        mlir::StringAttr::get(getContext(), "__dict__"), getDict()));

  vector.emplace_back(mlir::NamedAttribute(
      mlir::StringAttr::get(getContext(), "__kwdefaults__"), getKwDefaults()));
  vector.emplace_back(mlir::NamedAttribute(
      mlir::StringAttr::get(getContext(), "__qualname__"), getQualName()));
  return mlir::DictionaryAttr::get(getContext(), vector);
}

void pylir::Py::PylirPyDialect::initializeAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"
      >();
  addGlobalValueAttrInterfaces<
#define GEN_WRAP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyWrapInterfaces.h.inc"
      >(getContext());
}

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"
