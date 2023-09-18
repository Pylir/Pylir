//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyAttributes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <utility>

#include "PylirPyDialect.hpp"
#include "PylirPyOps.hpp"
#include "Value.hpp"

using namespace mlir;
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
    return {identity.cast<mlir::FlatSymbolRefAttr>()};
  }

  mlir::SymbolRefAttr identity;
  mlir::Operation* value{};
};

struct GlobalValueAttrStorage : mlir::AttributeStorage {
  using KeyTy = std::tuple<llvm::StringRef, bool, mlir::Attribute>;

  explicit GlobalValueAttrStorage(const KeyTy& key)
      : name(std::get<llvm::StringRef>(key)), constant(std::get<bool>(key)),
        initializer(std::get<mlir::Attribute>(key)) {}

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
            std::get<bool>(key), std::get<mlir::Attribute>(key)));
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
                             ConcreteObjectAttrInterface attribute) {
    initializer = attribute;
    return mlir::success();
  }

  llvm::StringRef name;
  bool constant;
  ConcreteObjectAttrInterface initializer;
};

} // namespace pylir::Py::detail

namespace {

#define GEN_WRAP_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyWrapInterfaces.h.inc"

template <class Interface>
struct RefAttrWrapInterface
    : WrapInterface<RefAttrWrapInterface<Interface>, Interface> {
  /// Returns the underlying instance that all interface methods are forwarded
  /// to by default.
  Interface getUnderlying(Attribute attribute) const {
    return cast<Interface>(
        cast<RefAttr>(attribute).getSymbol().getInitializerAttr());
  }

  /// Returns true if the `RefAttr` implements `Interface`.
  bool canImplement(Attribute thisAttr, std::in_place_type_t<Interface>) const {
    RefAttr refAttr = cast<RefAttr>(thisAttr);
    // Guard against not yet linked `RefAttr`s.
    if (!refAttr.getSymbol())
      return false;

    // If the interface inherits from `ConstObjectAttrInterface` it has an
    // implicit conversion to it and it is known that the symbol has to be
    // constant for the cast to be valid.
    if constexpr (std::is_convertible_v<Interface, ConstObjectAttrInterface>)
      if (!refAttr.getSymbol().getConstant())
        return false;

    // `RefAttr` without symbols never implements any interface, not even
    // `ObjectAttrInterface`.
    return isa_and_nonnull<Interface>(refAttr.getSymbol().getInitializerAttr());
  }
};

} // namespace

template <class... Interfaces>
static void addRefAttrInterfaces(MLIRContext* context) {
  RefAttr::attachInterface<RefAttrWrapInterface<Interfaces>...>(*context);
}

mlir::FlatSymbolRefAttr pylir::Py::RefAttr::getRef() const {
  return getImpl()->identity.cast<mlir::FlatSymbolRefAttr>();
}

pylir::Py::GlobalValueOp pylir::Py::RefAttr::getSymbol() const {
  return mlir::dyn_cast_or_null<GlobalValueOp>(getImpl()->value);
}

/// global-value ::= `#py.globalValue` `<` name { (`,` `const`)? (`,`
/// `initializer` `=` attr) }`>`
mlir::Attribute pylir::Py::GlobalValueAttr::parse(::mlir::AsmParser& parser,
                                                  ::mlir::Type) {
  // Keep a thread local stack to know whether we are printing any nested
  // `#py.globalValue`. Nested occurrences need to not print the initializer to
  // break any potential cycles.
  // TODO: Upstream should add support for this pattern.
  thread_local llvm::SetVector<llvm::StringRef> seenGlobalValueAttr;

  std::string name;
  if (parser.parseLess() || parser.parseKeywordOrString(&name))
    return nullptr;

  GlobalValueAttr attr = get(parser.getContext(), name);

  if (!seenGlobalValueAttr.insert(name)) {
    if (parser.parseGreater())
      return nullptr;

    return attr;
  }

  auto exit = llvm::make_scope_exit([&] { seenGlobalValueAttr.pop_back(); });

  // Default values.
  bool constant = false;
  ConcreteObjectAttrInterface initializer = {};

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

void pylir::Py::GlobalValueAttr::print(::mlir::AsmPrinter& printer) const {
  thread_local llvm::SetVector<GlobalValueAttr> seenGlobalValueAttr;

  printer << '<';

  printer.printKeywordOrString(getName());

  // Break a potential cycle by not printing a nested `#py.globalValue`.
  if (!seenGlobalValueAttr.insert(*this)) {
    printer << '>';
    return;
  }

  auto exit = llvm::make_scope_exit([&] { seenGlobalValueAttr.pop_back(); });

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

pylir::Py::ConcreteObjectAttrInterface
pylir::Py::GlobalValueAttr::getInitializer() const {
  return getImpl()->initializer;
}

void pylir::Py::GlobalValueAttr::setInitializer(
    ConcreteObjectAttrInterface initializer) {
  (void)Base::mutate(initializer);
}

void pylir::Py::PylirPyDialect::initializeAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"
      >();
  addRefAttrInterfaces<
#define GEN_WRAP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyWrapInterfaces.h.inc"
      >(getContext());
}

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

mlir::Attribute noRefCastNormalize(mlir::Attribute attr) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(attr)
      .Case([](pylir::Py::IntAttr intAttr) {
        return pylir::Py::FractionalAttr::get(
            intAttr.getContext(), intAttr.getValue(), pylir::BigInt(1));
      })
      .Case([](pylir::Py::FloatAttr floatAttr) {
        auto [nom, denom] = pylir::toRatio(floatAttr.getDoubleValue());
        return pylir::Py::FractionalAttr::get(floatAttr.getContext(),
                                              std::move(nom), std::move(denom));
      })
      .Case([](pylir::Py::StrAttr strAttr) {
        return mlir::StringAttr::get(strAttr.getContext(), strAttr.getValue());
      })
      .Default(attr);
}

UniqueOutput unique(llvm::ArrayRef<pylir::Py::DictAttr::Entry> entries) {
  // Doing this step ahead of time with a MapVector purely to get the correct
  // size ahead of time. Changing size would require rehashing.
  llvm::MapVector<mlir::Attribute, std::pair<mlir::Attribute, mlir::Attribute>>
      map;
  for (auto [key, normKey, value] : entries) {
    if (!normKey)
      normKey = noRefCastNormalize(key);

    auto [iter, inserted] = map.insert({normKey, {key, value}});
    if (!inserted)
      // We have to retain the key, but update the value it is mapped to.
      iter->second.second = value;
  }

  // Our Hashtable has a load factor of 0.9 currently. The tradeoff here is
  // between memory usage and lookup speed, especially in the case of a key not
  // being within the dictionary.
  UniqueOutput result;
  result.normalizedKeysUnique.resize(10 * map.size() / 9);
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
                                             llvm::ArrayRef<Entry> entries,
                                             RefAttr typeObject,
                                             mlir::DictionaryAttr slots) {
  typeObject =
      typeObject ? typeObject : RefAttr::get(context, Builtins::Dict.name);
  slots = slots ? slots : mlir::DictionaryAttr::get(context, {});

  auto result = ::unique(entries);
  return Base::get(context, result.normalizedKeysUnique,
                   result.keyValuePairsUnique, typeObject, slots);
}

mlir::Attribute pylir::Py::DictAttr::lookup(mlir::Attribute key) const {
  if (getNormalizedKeysInternal().empty())
    return nullptr;

  auto index =
      ::lookup(getNormalizedKeysInternal(), getCanonicalEqualsForm(key));
  const auto& entry = getNormalizedKeysInternal()[index];
  if (!entry.first)
    return nullptr;

  return getKeyValuePairs()[entry.second].second;
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
        mlir::Attribute key;
        mlir::Attribute normKey;
        mlir::Attribute value;
        if (parser.parseAttribute(key))
          return mlir::failure();

        if (mlir::succeeded(parser.parseOptionalKeyword("norm"))) {
          if (mlir::failed(parser.parseAttribute(normKey)))
            return mlir::failure();
        } else {
          normKey = noRefCastNormalize(key);
        }
        if (parser.parseKeyword("to") || parser.parseAttribute(value))
          return mlir::failure();

        entries.emplace_back(key, normKey, value);
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
  llvm::interleaveComma(values, printer.getStream(), [&](auto pair) {
    printer << pair.first;
    if (auto canonical = pylir::Py::getCanonicalEqualsForm(pair.first);
        canonical != noRefCastNormalize(pair.first))
      printer << " norm " << canonical;

    printer << " to " << pair.second;
  });
  printer << "}";
}

} // namespace

TypeAttrInterface pylir::Py::FunctionAttr::getTypeObject() const {
  return llvm::cast<TypeAttrInterface>(
      RefAttr::get(getContext(), Builtins::Function.name));
}

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

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"
