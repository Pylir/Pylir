#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>
#include <pylir/Support/Variant.hpp>

#include "PylirPyAttributes.hpp"

namespace
{
bool parseIterArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& operands,
                        mlir::ArrayAttr& iterExpansion)
{
    llvm::SmallVector<std::int32_t> iters;
    auto exit = llvm::make_scope_exit([&] { iterExpansion = parser.getBuilder().getI32ArrayAttr(iters); });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]
    {
        if (!parser.parseOptionalStar())
        {
            iters.push_back(index);
        }
        index++;
        return parser.parseOperand(operands.emplace_back());
    };
    if (parseOnce())
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (parseOnce())
        {
            return true;
        }
    }

    if (parser.parseRParen())
    {
        return true;
    }

    return false;
}

void printIterArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange operands,
                        mlir::ArrayAttr iterExpansion)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : iterExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    int i = 0;
    llvm::interleaveComma(operands, printer,
                          [&](mlir::Value value)
                          {
                              if (iters.contains(i))
                              {
                                  printer << '*' << value;
                              }
                              else
                              {
                                  printer << value;
                              }
                              i++;
                          });
    printer << ')';
}

bool parseMappingArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& keys,
                           llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& values,
                           mlir::ArrayAttr& mappingExpansion)
{
    llvm::SmallVector<std::int32_t> mappings;
    auto exit = llvm::make_scope_exit([&] { mappingExpansion = parser.getBuilder().getI32ArrayAttr(mappings); });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]() -> mlir::ParseResult
    {
        if (!parser.parseOptionalStar())
        {
            if (parser.parseStar())
            {
                return mlir::failure();
            }
            mappings.push_back(index);
            index++;
            return parser.parseOperand(keys.emplace_back());
        }
        index++;
        return mlir::failure(parser.parseOperand(keys.emplace_back()) || parser.parseColon()
                             || parser.parseOperand(values.emplace_back()));
    };
    if (parseOnce())
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (parseOnce())
        {
            return true;
        }
    }

    if (parser.parseRParen())
    {
        return true;
    }

    return false;
}

void printMappingArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange keys,
                           mlir::OperandRange values, mlir::ArrayAttr mappingExpansion)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : mappingExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    int i = 0;
    std::size_t valueCounter = 0;
    llvm::interleaveComma(keys, printer,
                          [&](mlir::Value key)
                          {
                              if (iters.contains(i))
                              {
                                  printer << "**" << key;
                                  i++;
                                  return;
                              }
                              printer << key << " : " << values[valueCounter++];
                              i++;
                          });
    printer << ')';
}

mlir::Attribute toBool(mlir::Attribute value)
{
    return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(value).Default({});
}

bool isStrictTuple(mlir::Value value)
{
    if (value.getDefiningOp<pylir::Py::MakeTupleOp>())
    {
        return true;
    }
    auto constant = value.getDefiningOp<pylir::Py::ConstantOp>();
    if (!constant)
    {
        return false;
    }
    return constant.constant().isa<pylir::Py::TupleAttr>();
}

bool isStrictDict(mlir::Value value)
{
    if (value.getDefiningOp<pylir::Py::MakeDictOp>())
    {
        return true;
    }
    auto constant = value.getDefiningOp<pylir::Py::ConstantOp>();
    if (!constant)
    {
        return false;
    }
    return constant.constant().isa<pylir::Py::DictAttr>();
}

} // namespace

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return constant();
}

mlir::OpFoldResult pylir::Py::MakeTupleOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!std::all_of(operands.begin(), operands.end(),
                     [](mlir::Attribute attr) -> bool { return static_cast<bool>(attr); }))
    {
        return nullptr;
    }
    llvm::SmallVector<mlir::Attribute> result;
    auto expanders = llvm::to_vector<4>(iterExpansion().getAsValueRange<mlir::IntegerAttr>());
    std::reverse(expanders.begin(), expanders.end());
    for (auto iter : llvm::enumerate(operands))
    {
        if (expanders.empty() || expanders.back() != iter.index())
        {
            result.push_back(iter.value());
            continue;
        }
        expanders.pop_back();
        if (!llvm::TypeSwitch<mlir::Attribute, bool>(iter.value())
                 .Case<Py::ListAttr, Py::TupleAttr, Py::SetAttr>(
                     [&](auto sequences)
                     {
                         result.insert(result.end(), sequences.getValue().begin(), sequences.getValue().end());
                         return true;
                     })
                 .Case(
                     [&](Py::DictAttr dictAttr)
                     {
                         auto second = llvm::make_second_range(dictAttr.getValue());
                         result.insert(result.end(), second.begin(), second.end());
                         return true;
                     })
                 .Case(
                     [&](mlir::StringAttr stringAttr)
                     {
                         auto utf32 = Text::toUTF32String(stringAttr.getValue());
                         auto mapped = llvm::map_range(
                             utf32,
                             [&](char32_t codepoint) {
                                 return mlir::StringAttr::get(getContext(), Text::toUTF8String({&codepoint, 1}));
                             });
                         result.insert(result.end(), mapped.begin(), mapped.end());
                         return true;
                     })
                 .Default(false))
        {
            return nullptr;
        }
    }
    return Py::TupleAttr::get(getContext(), result);
}

namespace
{
void commonType(mlir::Attribute& lhs, mlir::Attribute& rhs)
{
    if (lhs.dyn_cast_or_null<mlir::FloatAttr>())
    {
        if (auto integer = rhs.dyn_cast_or_null<pylir::Py::IntAttr>())
        {
            rhs = mlir::FloatAttr::get(mlir::Float64Type::get(rhs.getContext()), integer.getValue().roundToDouble());
        }
    }
    if (rhs.dyn_cast_or_null<mlir::FloatAttr>())
    {
        if (auto integer = lhs.dyn_cast_or_null<pylir::Py::IntAttr>())
        {
            lhs = mlir::FloatAttr::get(mlir::Float64Type::get(lhs.getContext()), integer.getValue().roundToDouble());
        }
    }
}
} // namespace

mlir::OpFoldResult pylir::Py::PowerOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    // I need both operands, even if some identities only need the exponent or base, because this might just fold to
    // a type error instead and the type returned is dependent on the value and type of each.
    if (!operands[0] || !operands[1])
    {
        return nullptr;
    }

    // TODO: fold with mod
    if (mod())
    {
        return nullptr;
    }
    auto base = operands[0];
    auto exponent = operands[1];
    commonType(base, exponent);
    if (base.getTypeID() != exponent.getTypeID())
    {
        return nullptr;
    }
    // Negative exponent causes the result and operands to be converted to double
    if (auto integer = exponent.dyn_cast_or_null<Py::IntAttr>(); integer && integer.getValue().isNegative())
    {
        exponent =
            mlir::FloatAttr::get(mlir::Float64Type::get(exponent.getContext()), integer.getValue().roundToDouble());
        base = mlir::FloatAttr::get(mlir::Float64Type::get(base.getContext()),
                                    base.cast<Py::IntAttr>().getValue().roundToDouble());
    }
    // If the exponent is 1, return the base
    if (auto integer = exponent.dyn_cast_or_null<Py::IntAttr>(); integer && integer.getValue() == BigInt(1))
    {
        return base;
    }
    if (auto floating = exponent.dyn_cast_or_null<mlir::FloatAttr>();
        floating && floating.getValue().convertToDouble() == 1)
    {
        return base;
    }

    // If the base is 1, return 1
    if (auto integer = base.dyn_cast_or_null<Py::IntAttr>(); integer && integer.getValue() == BigInt(1))
    {
        return base;
    }
    if (auto floating = base.dyn_cast_or_null<mlir::FloatAttr>();
        floating && floating.getValue().convertToDouble() == 1)
    {
        return base;
    }
    if (base.isa<mlir::FloatAttr>())
    {
        auto result = std::pow(base.cast<mlir::FloatAttr>().getValue().convertToDouble(),
                               exponent.cast<mlir::FloatAttr>().getValue().convertToDouble());
        return mlir::FloatAttr::get(mlir::Float64Type::get(base.getContext()), result);
    }
    if (base.isa<Py::IntAttr>())
    {
        auto expo = exponent.cast<Py::IntAttr>().getValue().tryGetInteger<int>();
        if (!expo)
        {
            return nullptr;
        }
        auto result = pow(base.cast<Py::IntAttr>().getValue(), *expo);
        return Py::IntAttr::get(getContext(), std::move(result));
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::NegOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!operands[0])
    {
        return nullptr;
    }
    if (auto floating = operands[0].dyn_cast_or_null<mlir::FloatAttr>())
    {
        return mlir::FloatAttr::get(floating.getType(), llvm::neg(floating.getValue()));
    }
    if (auto integer = operands[0].dyn_cast_or_null<Py::IntAttr>())
    {
        return Py::IntAttr::get(getContext(), -integer.getValue());
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::PosOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!operands[0])
    {
        return nullptr;
    }
    if (operands[0].isa<mlir::FloatAttr, Py::IntAttr>())
    {
        return operands[0];
    }
    return nullptr;
}

mlir::OpFoldResult pylir::Py::InvertOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!operands[0])
    {
        return nullptr;
    }
    if (auto integer = operands[0].dyn_cast_or_null<Py::IntAttr>())
    {
        return Py::IntAttr::get(getContext(), ~integer.getValue());
    }
    return nullptr;
}

namespace
{
template <class... Bins>
mlir::OpFoldResult foldBin(mlir::Attribute lhs, mlir::Attribute rhs, Bins... bins)
{
    if (!lhs && !rhs)
    {
        return nullptr;
    }
    auto typeSwitch = llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(lhs);
    (
        [&](auto bin)
        {
            using T = std::decay_t<decltype(bin)>;
            using TTrait = llvm::function_traits<T>;
            static_assert(TTrait::num_args == 2);
            typeSwitch.Case(
                [&](typename TTrait::template arg_t<0> first)
                {
                    auto subSwitch = llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(rhs);
                    (
                        [&](auto searchedBin)
                        {
                            using U = std::decay_t<decltype(searchedBin)>;
                            using UTrait = llvm::function_traits<U>;
                            if constexpr (std::is_convertible_v<typename TTrait::template arg_t<0>,
                                                                typename UTrait::template arg_t<0>>)
                            {
                                subSwitch.Case([&](typename UTrait::template arg_t<1> second)
                                               { return searchedBin(first, second); });
                            }
                        }(bins),
                        ...);
                    return subSwitch.Default(mlir::OpFoldResult(nullptr));
                });
        }(bins),
        ...);
    return typeSwitch.Default(mlir::OpFoldResult(nullptr));
}

constexpr auto arithmeticCase = [](mlir::Attribute lhs, mlir::Attribute rhs, auto binOp) -> mlir::OpFoldResult
{
    commonType(lhs, rhs);
    return llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(lhs)
        .Case(
            [&](::pylir::Py::IntAttr attr) -> mlir::OpFoldResult
            {
                return ::pylir::Py::IntAttr::get(attr.getContext(),
                                                 binOp(attr.getValue(), rhs.cast<::pylir::Py::IntAttr>().getValue()));
            })
        .Case(
            [&](mlir::FloatAttr attr) -> mlir::OpFoldResult
            {
                return mlir::FloatAttr::get(
                    attr.getType(),
                    binOp(attr.getValue().convertToDouble(), rhs.cast<mlir::FloatAttr>().getValue().convertToDouble()));
            })
        .Default(mlir::OpFoldResult(nullptr));
};
} // namespace

mlir::OpFoldResult pylir::Py::MulOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [](Py::IntAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::multiplies{}); },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::multiplies{}); },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::multiplies{}); },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::multiplies{}); });
}

mlir::OpFoldResult pylir::Py::FloorDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::IntAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return Py::IntAttr::get(getContext(), lhs.getValue() / rhs.getValue());
        },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(), std::floor(lhs.getValueAsDouble() / rhs.getValueAsDouble()));
        },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(rhs.getType(),
                                        std::floor(lhs.getValue().roundToDouble() / rhs.getValueAsDouble()));
        },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(),
                                        std::floor(lhs.getValueAsDouble() / rhs.getValue().roundToDouble()));
        });
}

mlir::OpFoldResult pylir::Py::TrueDivOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::IntAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(mlir::Float64Type::get(getContext()),
                                        lhs.getValue().roundToDouble() / rhs.getValue().roundToDouble());
        },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(), lhs.getValue() / rhs.getValue());
        },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(rhs.getType(), llvm::APFloat(lhs.getValue().roundToDouble()) / rhs.getValue());
        },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(), lhs.getValue() / llvm::APFloat(rhs.getValue().roundToDouble()));
        });
}

mlir::OpFoldResult pylir::Py::ModuloOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::IntAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return Py::IntAttr::get(getContext(), lhs.getValue() % rhs.getValue());
        },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(), std::fmod(lhs.getValueAsDouble(), rhs.getValueAsDouble()));
        },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(rhs.getType(),
                                        std::fmod(lhs.getValue().roundToDouble(), rhs.getValueAsDouble()));
        },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
        {
            if (rhs.getValue().isZero())
            {
                return nullptr;
            }
            return mlir::FloatAttr::get(lhs.getType(),
                                        std::fmod(lhs.getValueAsDouble(), rhs.getValue().roundToDouble()));
        });
}

mlir::OpFoldResult pylir::Py::AddOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [](Py::IntAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::plus{}); },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::plus{}); },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::plus{}); },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::plus{}); });
}

mlir::OpFoldResult pylir::Py::SubOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [](Py::IntAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::minus{}); },
        [](mlir::FloatAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::minus{}); },
        [](Py::IntAttr lhs, mlir::FloatAttr rhs) { return arithmeticCase(lhs, rhs, std::minus{}); },
        [](mlir::FloatAttr lhs, Py::IntAttr rhs) { return arithmeticCase(lhs, rhs, std::minus{}); });
}

mlir::OpFoldResult pylir::Py::LShiftOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(operands[0], operands[1],
                   [this](Py::IntAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
                   {
                       auto value = rhs.getValue().tryGetInteger<int>();
                       if (!value)
                       {
                           return nullptr;
                       }
                       return Py::IntAttr::get(getContext(), lhs.getValue() << *value);
                   });
}

mlir::OpFoldResult pylir::Py::RShiftOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(operands[0], operands[1],
                   [this](Py::IntAttr lhs, Py::IntAttr rhs) -> mlir::OpFoldResult
                   {
                       auto value = rhs.getValue().tryGetInteger<int>();
                       if (!value)
                       {
                           return nullptr;
                       }
                       return Py::IntAttr::get(getContext(), lhs.getValue() >> *value);
                   });
}

mlir::OpFoldResult pylir::Py::AndOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::BoolAttr lhs, Py::BoolAttr rhs)
        { return Py::BoolAttr::get(getContext(), lhs.getValue() & rhs.getValue()); },
        [this](Py::IntAttr lhs, Py::IntAttr rhs)
        { return Py::IntAttr::get(getContext(), lhs.getValue() & rhs.getValue()); });
}

mlir::OpFoldResult pylir::Py::OrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::BoolAttr lhs, Py::BoolAttr rhs)
        { return Py::BoolAttr::get(getContext(), lhs.getValue() | rhs.getValue()); },
        [this](Py::IntAttr lhs, Py::IntAttr rhs)
        { return Py::IntAttr::get(getContext(), lhs.getValue() | rhs.getValue()); });
}

mlir::OpFoldResult pylir::Py::XorOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return foldBin(
        operands[0], operands[1],
        [this](Py::BoolAttr lhs, Py::BoolAttr rhs)
        { return Py::BoolAttr::get(getContext(), lhs.getValue() ^ rhs.getValue()); },
        [this](Py::IntAttr lhs, Py::IntAttr rhs)
        { return Py::IntAttr::get(getContext(), lhs.getValue() ^ rhs.getValue()); });
}

mlir::LogicalResult pylir::Py::MakeTupleOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeListOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                            ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                            ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                            ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeSetOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                           ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                           ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                           ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult pylir::Py::MakeDictOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                            ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                            ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                            ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::OpFoldResult pylir::Py::BoolOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (!operands[0])
    {
        return nullptr;
    }
    auto result = toBool(operands[0]);
    if (!result)
    {
        return nullptr;
    }
    return result;
}

mlir::OpFoldResult pylir::Py::BoolToI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto boolean = operands[0].dyn_cast_or_null<Py::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return mlir::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::BoolFromI1Op::fold(::llvm::ArrayRef<mlir::Attribute> operands)
{
    auto boolean = operands[0].dyn_cast_or_null<mlir::BoolAttr>();
    if (!boolean)
    {
        return nullptr;
    }
    return Py::BoolAttr::get(getContext(), boolean.getValue());
}

mlir::OpFoldResult pylir::Py::IsUnboundValueOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (operands[0])
    {
        return mlir::BoolAttr::get(getContext(), operands[0].isa<Py::UnboundAttr>());
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::GetGlobalValueOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<Py::GlobalValueOp>(*this, name()));
}

mlir::LogicalResult pylir::Py::GetGlobalHandleOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<Py::GlobalHandleOp>(*this, name()));
}

mlir::LogicalResult pylir::Py::MakeFuncOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, function()));
}

mlir::LogicalResult pylir::Py::MakeClassOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, initFunc()));
}

void pylir::Py::MakeTupleOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeListOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeSetOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                 const std::vector<::pylir::Py::IterArg>& args)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion));
}

void pylir::Py::MakeDictOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                  const std::vector<::pylir::Py::DictArg>& args)
{
    std::vector<mlir::Value> keys, values;
    std::vector<std::int32_t> mappingExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(),
            [&](std::pair<mlir::Value, mlir::Value> pair)
            {
                keys.push_back(pair.first);
                values.push_back(pair.second);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, values, odsBuilder.getI32ArrayAttr(mappingExpansion));
}

mlir::OpFoldResult pylir::Py::IsOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    {
        auto lhsGlobal = lhs().getDefiningOp<Py::GetGlobalValueOp>();
        auto rhsGlobal = rhs().getDefiningOp<Py::GetGlobalValueOp>();
        if (lhsGlobal && rhsGlobal)
        {
            return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1),
                                          rhsGlobal.name() == lhsGlobal.name());
        }
    }
    if (lhs() == rhs())
    {
        return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), true);
    }
    {
        auto lhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(lhs().getDefiningOp());
        auto rhsEffect = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(rhs().getDefiningOp());
        if (lhsEffect && rhsEffect && lhsEffect.hasEffect<mlir::MemoryEffects::Allocate>()
            && rhsEffect.hasEffect<mlir::MemoryEffects::Allocate>())
        {
            return mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 1), false);
        }
    }
    return nullptr;
}

mlir::LogicalResult pylir::Py::InvokeOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom<mlir::FuncOp>(*this, callee()));
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

mlir::CallInterfaceCallable pylir::Py::InvokeOp::getCallableForCallee()
{
    return calleeAttr();
}

mlir::Operation::operand_range pylir::Py::InvokeOp::getArgOperands()
{
    return operands();
}

mlir::LogicalResult pylir::Py::InvokeOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                          ::llvm::Optional<::mlir::Location> ,
                                                          ::mlir::ValueRange ,
                                                          ::mlir::DictionaryAttr ,
                                                          ::mlir::RegionRange ,
                                                          ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::InvokeIndirectOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

mlir::CallInterfaceCallable pylir::Py::InvokeIndirectOp::getCallableForCallee()
{
    return callee();
}

mlir::Operation::operand_range pylir::Py::InvokeIndirectOp::getArgOperands()
{
    return operands();
}

mlir::LogicalResult
    pylir::Py::InvokeIndirectOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                                  ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                  ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::LogicalResult
    pylir::Py::MakeTupleExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                               ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                               ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeTupleExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeTupleExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                     const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                     mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                     mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult
    pylir::Py::MakeListExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeListExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeListExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult pylir::Py::MakeSetExOp::inferReturnTypes(::mlir::MLIRContext* context,
                                                             ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange,
                                                             ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                                             ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeSetExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeSetExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   const std::vector<::pylir::Py::IterArg>& args, mlir::Block* happyPath,
                                   mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                   mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> values;
    std::vector<std::int32_t> iterExpansion;
    for (auto& iter : llvm::enumerate(args))
    {
        pylir::match(
            iter.value(), [&](mlir::Value value) { values.push_back(value); },
            [&](Py::IterExpansion expansion)
            {
                values.push_back(expansion.value);
                iterExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, values, odsBuilder.getI32ArrayAttr(iterExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

mlir::LogicalResult
    pylir::Py::MakeDictExOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    inferredReturnTypes.push_back(Py::DynamicType::get(context));
    return mlir::success();
}

mlir::Optional<mlir::MutableOperandRange> pylir::Py::MakeDictExOp::getMutableSuccessorOperands(unsigned int index)
{
    if (index == 0)
    {
        return normalDestOperandsMutable();
    }
    return llvm::None;
}

void pylir::Py::MakeDictExOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    const std::vector<::pylir::Py::DictArg>& keyValues, mlir::Block* happyPath,
                                    mlir::ValueRange normalDestOperands, mlir::Block* unwindPath,
                                    mlir::ValueRange unwindDestOperands)
{
    std::vector<mlir::Value> keys, values;
    std::vector<std::int32_t> mappingExpansion;
    for (auto& iter : llvm::enumerate(keyValues))
    {
        pylir::match(
            iter.value(),
            [&](std::pair<mlir::Value, mlir::Value> pair)
            {
                keys.push_back(pair.first);
                values.push_back(pair.second);
            },
            [&](Py::MappingExpansion expansion)
            {
                keys.push_back(expansion.value);
                mappingExpansion.push_back(iter.index());
            });
    }
    build(odsBuilder, odsState, keys, values, odsBuilder.getI32ArrayAttr(mappingExpansion), normalDestOperands,
          unwindDestOperands, happyPath, unwindPath);
}

namespace
{
mlir::LogicalResult verify(pylir::Py::ConstantOp op)
{
    for (auto& uses : op->getUses())
    {
        if (auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(uses.getOwner());
            interface && interface.getEffectOnValue<mlir::MemoryEffects::Write>(op))
        {
            return uses.getOwner()->emitError("Write to a constant value is not allowed");
        }
    }
    return mlir::success();
}
} // namespace

#include <pylir/Optimizer/PylirPy/IR/PylirPyOpsEnums.cpp.inc>

// TODO remove MLIR 14
using namespace mlir;

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
