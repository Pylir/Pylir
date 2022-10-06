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

template <>
struct mlir::FieldParser<llvm::APFloat>
{
    static mlir::FailureOr<llvm::APFloat> parse(mlir::AsmParser& parser)
    {
        double value;
        if (parser.parseFloat(value))
        {
            return mlir::failure();
        }
        return llvm::APFloat(value);
    }
};

template <>
struct mlir::FieldParser<pylir::BigInt>
{
    static mlir::FailureOr<pylir::BigInt> parse(mlir::AsmParser& parser)
    {
        llvm::APInt apInt;
        if (parser.parseInteger(apInt))
        {
            return mlir::failure();
        }
        llvm::SmallString<10> str;
        apInt.toStringSigned(str);
        return pylir::BigInt(std::string{str.data(), str.size()});
    }
};

namespace pylir
{
llvm::hash_code hash_value(const pylir::BigInt& bigInt)
{
    auto count = mp_sbin_size(&bigInt.getHandle());
    llvm::SmallVector<std::uint8_t, 10> data(count);
    auto result = mp_to_sbin(&bigInt.getHandle(), data.data(), count, nullptr);
    PYLIR_ASSERT(result == MP_OKAY);
    return llvm::hash_value(makeArrayRef(data));
}

namespace Py::detail
{
struct RefAttrStorage : mlir::AttributeStorage
{
    using KeyTy = std::tuple<mlir::FlatSymbolRefAttr>;

    explicit RefAttrStorage(mlir::FlatSymbolRefAttr identity) : identity(identity) {}

    bool operator==(const KeyTy& key) const
    {
        return std::get<0>(key) == identity;
    }

    static RefAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<RefAttrStorage>()) RefAttrStorage(std::get<0>(key));
    }

    mlir::SymbolRefAttr identity;
    mlir::Operation* value{};
};
} // namespace Py::detail
} // namespace pylir

mlir::FlatSymbolRefAttr pylir::Py::RefAttr::getRef() const
{
    return getImpl()->identity.cast<mlir::FlatSymbolRefAttr>();
}

pylir::Py::GlobalValueOp pylir::Py::RefAttr::getSymbol() const
{
    return mlir::dyn_cast_or_null<GlobalValueOp>(getImpl()->value);
}

void pylir::Py::PylirPyDialect::initializeAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"
        >();
}

namespace
{

std::size_t lookup(llvm::ArrayRef<std::pair<mlir::Attribute, std::size_t>> normalizedKeys, mlir::Attribute key)
{
    for (auto bucket = hash_value(key) % normalizedKeys.size();;
         // This is essentially bucket = (bucket + 1) % normalizedKeys.size(), but uses a conditional move instead
         // of a module operation on each iteration.
         bucket = bucket + 1 < normalizedKeys.size() ? bucket + 1 : 0)
    {
        if (!normalizedKeys[bucket].first || key == normalizedKeys[bucket].first)
        {
            return bucket;
        }
    }
    PYLIR_UNREACHABLE;
}

struct UniqueOutput
{
    llvm::SmallVector<std::pair<mlir::Attribute, std::size_t>> normalizedKeysUnique;
    llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> keyValuePairsUnique;
};

mlir::Attribute noRefCastNormalize(mlir::Attribute attr)
{
    return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(attr)
        .Case([](pylir::Py::IntAttr intAttr)
              { return pylir::Py::FractionalAttr::get(intAttr.getContext(), intAttr.getValue(), pylir::BigInt(1)); })
        .Case(
            [](pylir::Py::FloatAttr floatAttr)
            {
                auto [nom, denom] = pylir::toRatio(floatAttr.getDoubleValue());
                return pylir::Py::FractionalAttr::get(floatAttr.getContext(), std::move(nom), std::move(denom));
            })
        .Case([](pylir::Py::StrAttr strAttr)
              { return mlir::StringAttr::get(strAttr.getContext(), strAttr.getValue()); })
        .Default(attr);
}

UniqueOutput unique(llvm::ArrayRef<pylir::Py::DictAttr::Entry> entries)
{
    // Doing this step ahead of time with a MapVector purely to get the correct size ahead of time. Changing size
    // would require rehashing.
    llvm::MapVector<mlir::Attribute, std::pair<mlir::Attribute, mlir::Attribute>> map;
    for (auto [key, normKey, value] : entries)
    {
        if (!normKey)
        {
            normKey = noRefCastNormalize(key);
        }
        auto [iter, inserted] = map.insert({normKey, {key, value}});
        if (!inserted)
        {
            // We have to retain the key, but update the value it is mapped to.
            iter->second.second = value;
        }
    }

    // Our Hashtable has a load factor of 0.9 currently. The tradeoff here is between memory usage and lookup
    // speed, especially in the case of a key not being within the dictionary.
    UniqueOutput result;
    result.normalizedKeysUnique.resize(10 * map.size() / 9);
    result.keyValuePairsUnique.reserve(map.size());
    for (auto& [key, pair] : map)
    {
        result.normalizedKeysUnique[lookup(result.normalizedKeysUnique, key)] = {key,
                                                                                 result.keyValuePairsUnique.size()};
        result.keyValuePairsUnique.push_back(std::move(pair));
    }
    return result;
}

} // namespace

pylir::Py::DictAttr pylir::Py::DictAttr::get(mlir::MLIRContext* context, llvm::ArrayRef<Entry> entries,
                                             RefAttr typeObject, mlir::DictionaryAttr slots)
{
    typeObject = typeObject ? typeObject : RefAttr::get(context, Builtins::Dict.name);
    slots = slots ? slots : mlir::DictionaryAttr::get(context, {});

    auto result = ::unique(entries);
    return Base::get(context, result.normalizedKeysUnique, result.keyValuePairsUnique, typeObject, slots);
}

mlir::Attribute pylir::Py::DictAttr::lookup(mlir::Attribute key) const
{
    if (getNormalizedKeysInternal().empty())
    {
        return nullptr;
    }
    auto index = ::lookup(getNormalizedKeysInternal(), getCanonicalEqualsForm(key));
    const auto& entry = getNormalizedKeysInternal()[index];
    if (!entry.first)
    {
        return nullptr;
    }
    return getKeyValuePairs()[entry.second].second;
}

namespace
{

mlir::LogicalResult
    parseKVPair(mlir::AsmParser& parser,
                mlir::FailureOr<llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>>>& values,
                mlir::FailureOr<llvm::SmallVector<std::pair<mlir::Attribute, std::size_t>>>& normalizedKeys)
{
    values.emplace();
    llvm::SmallVector<pylir::Py::DictAttr::Entry> entries;
    auto parseResult = parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces,
                                                      [&]
                                                      {
                                                          mlir::Attribute key;
                                                          mlir::Attribute normKey;
                                                          mlir::Attribute value;
                                                          if (parser.parseAttribute(key))
                                                          {
                                                              return mlir::failure();
                                                          }
                                                          if (mlir::succeeded(parser.parseOptionalKeyword("norm")))
                                                          {
                                                              if (mlir::failed(parser.parseAttribute(normKey)))
                                                              {
                                                                  return mlir::failure();
                                                              }
                                                          }
                                                          else
                                                          {
                                                              normKey = noRefCastNormalize(key);
                                                          }
                                                          if (parser.parseKeyword("to") || parser.parseAttribute(value))
                                                          {
                                                              return mlir::failure();
                                                          }
                                                          entries.emplace_back(key, normKey, value);
                                                          return mlir::success();
                                                      });
    if (mlir::failed(parseResult))
    {
        return mlir::failure();
    }

    auto result = ::unique(entries);
    values = std::move(result.keyValuePairsUnique);
    normalizedKeys = std::move(result.normalizedKeysUnique);

    return mlir::success();
}

void printKVPair(mlir::AsmPrinter& printer, llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> values,
                 llvm::ArrayRef<std::pair<mlir::Attribute, std::size_t>>)
{
    printer << "{";
    llvm::interleaveComma(values, printer.getStream(),
                          [&](auto pair)
                          {
                              printer << pair.first;
                              if (auto canonical = pylir::Py::getCanonicalEqualsForm(pair.first);
                                  canonical != noRefCastNormalize(pair.first))
                              {
                                  printer << " norm " << canonical;
                              }
                              printer << " to " << pair.second;
                          });
    printer << "}";
}

} // namespace

void pylir::Py::BoolAttr::print(mlir::AsmPrinter& printer) const
{
    printer << "<" << (getValue() ? "True" : "False") << ">";
}

mlir::Attribute pylir::Py::BoolAttr::parse(mlir::AsmParser& parser, mlir::Type)
{
    llvm::StringRef keyword;
    llvm::SMLoc loc;
    if (parser.parseLess() || parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword) || parser.parseGreater())
    {
        return {};
    }
    if (keyword != "True" && keyword != "False")
    {
        parser.emitError(loc, "Expected one of 'True' or 'False'");
        return {};
    }
    return get(parser.getContext(), keyword == "True");
}

pylir::Py::RefAttr pylir::Py::FunctionAttr::getTypeObject() const
{
    return RefAttr::get(getContext(), Builtins::Function.name);
}

mlir::DictionaryAttr pylir::Py::FunctionAttr::getSlots() const
{
    llvm::SmallVector<mlir::NamedAttribute> vector = {
        mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__defaults__"), getDefaults()),
    };
    if (getDict())
    {
        vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__dict__"), getDict()));
    }
    vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__kwdefaults__"), getKwDefaults()));
    vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__qualname__"), getQualName()));
    return mlir::DictionaryAttr::get(getContext(), vector);
}

void pylir::Py::DictAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    llvm::for_each(getNormalizedKeys(), walkAttrsFn);
    llvm::for_each(getKeyValuePairs(),
                   [&](auto&& pair)
                   {
                       walkAttrsFn(pair.first);
                       walkAttrsFn(pair.second);
                   });
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::DictAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto normalizedKeys = replAttrs.take_front(getKeyValuePairs().size());
    replAttrs = replAttrs.drop_front(getKeyValuePairs().size());
    std::vector<Entry> vector;
    for (std::size_t i = 0; i < replAttrs.size() - 2; i += 2)
    {
        vector.emplace_back(replAttrs[i], nullptr, replAttrs[i + 1]);
    }

    for (auto [newKey, oldEntry] :
         llvm::zip(normalizedKeys, llvm::make_filter_range(
                                       getNormalizedKeysInternal(),
                                       +[](std::pair<mlir::Attribute, std::size_t> pair) { return pair.first; })))
    {
        vector[oldEntry.second].normalizedKey = newKey;
    }

    auto type = replAttrs.take_back(2).back().cast<RefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return get(getContext(), vector, type, slots);
}

namespace
{
template <class Op>
void doTypeObjectSlotsWalk(Op op, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn)
{
    walkAttrsFn(op.getTypeObject());
    walkAttrsFn(op.getSlots());
}

template <class Op, class... Args>
Op doTypeObjectSlotsReplace(Op op, llvm::ArrayRef<mlir::Attribute> replAttrs, Args&&... prior)
{
    auto type = replAttrs.take_back(2).back().cast<pylir::Py::RefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return Op::get(op.getContext(), std::forward<Args>(prior)..., type, slots);
}
} // namespace

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::ObjectAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                   llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs);
}

void pylir::Py::IntAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::IntAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::FloatAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::FloatAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                  llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::StrAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::StrAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::TupleAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    for (const auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
}

mlir::Attribute pylir::Py::TupleAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                  llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.back().cast<RefAttr>();
    return get(getContext(), replAttrs.drop_back(), typeObject);
}

void pylir::Py::ListAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    for (const auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::ListAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.take_back(2).front().cast<RefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return get(getContext(), replAttrs.drop_back(2), typeObject, slots);
}

void pylir::Py::SetAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    for (const auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::SetAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.take_back(2).front().cast<RefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return get(getContext(), replAttrs.drop_back(2), typeObject, slots);
}

void pylir::Py::FunctionAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                       llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getValue());
    walkAttrsFn(getQualName());
    walkAttrsFn(getDefaults());
    walkAttrsFn(getKwDefaults());
    if (getDict())
    {
        walkAttrsFn(getDict());
    }
}

mlir::Attribute pylir::Py::FunctionAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                     llvm::ArrayRef<mlir::Type>) const
{
    auto value = replAttrs[0].cast<mlir::FlatSymbolRefAttr>();
    auto qualName = replAttrs[1];
    auto defaults = replAttrs[2];
    auto kwDefaults = replAttrs[4];
    auto dict = replAttrs.size() > 5 ? replAttrs[5] : getDict();
    return get(getContext(), value, qualName, defaults, kwDefaults, dict);
}

void pylir::Py::TypeAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getMroTuple());
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::TypeAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto value = replAttrs[0];
    auto typeObject = replAttrs[1].cast<RefAttr>();
    auto slots = replAttrs[2].cast<mlir::DictionaryAttr>();
    return get(getContext(), value, typeObject, slots);
}

void pylir::Py::RefAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getRef());
}

mlir::Attribute pylir::Py::RefAttr::replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs,
                                                                ::llvm::ArrayRef<::mlir::Type>) const
{
    return RefAttr::get(getContext(), replAttrs[0].cast<mlir::FlatSymbolRefAttr>());
}

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.cpp.inc"

void pylir::Py::PylirPyDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& os) const
{
    if (auto boolAttr = attr.dyn_cast<BoolAttr>())
    {
        os << BoolAttr::getMnemonic();
        boolAttr.print(os);
        return;
    }
    (void)generatedAttributePrinter(attr, os);
}

mlir::Attribute pylir::Py::PylirPyDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const
{
    llvm::StringRef keyword;
    mlir::Attribute res;
    auto loc = parser.getCurrentLocation();
    if (auto opt = generatedAttributeParser(parser, &keyword, type, res); opt.has_value())
    {
        if (mlir::failed(*opt))
        {
            return {};
        }
        return res;
    }
    if (keyword == BoolAttr::getMnemonic())
    {
        return BoolAttr::parse(parser, type);
    }
    parser.emitError(loc, "Unknown dialect attribute: ") << keyword;
    return res;
}
