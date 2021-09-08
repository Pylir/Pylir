#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>

#include "PylirPyAttributes.hpp"

namespace
{
bool parseExpandArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& operands,
                          mlir::ArrayAttr& iterExpansion, mlir::ArrayAttr& mappingExpansion,
                          mlir::ArrayAttr& keywordArgs)
{
    llvm::SmallVector<std::int32_t> iters;
    llvm::SmallVector<std::int32_t> mappings;
    llvm::SmallVector<llvm::StringRef> keywords;
    auto exit = llvm::make_scope_exit(
        [&]
        {
            iterExpansion = parser.getBuilder().getI32ArrayAttr(iters);
            mappingExpansion = parser.getBuilder().getI32ArrayAttr(mappings);
            keywordArgs = parser.getBuilder().getStrArrayAttr(keywords);
        });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]() -> mlir::LogicalResult
    {
        llvm::StringRef keyword;
        if (!parser.parseOptionalStar())
        {
            if (!parser.parseOptionalStar())
            {
                mappings.push_back(index);
            }
            else
            {
                iters.push_back(index);
            }
        }
        else if (!parser.parseOptionalKeyword(&keyword))
        {
            if (parser.parseEqual())
            {
                return mlir::failure();
            }
        }
        keywords.push_back(keyword);
        index++;
        return parser.parseOperand(operands.emplace_back());
    };
    if (mlir::failed(parseOnce()))
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (mlir::failed(parseOnce()))
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

void printExpandArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange operands,
                          mlir::ArrayAttr iterExpansion, mlir::ArrayAttr mappingExpansion, mlir::ArrayAttr keywords)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : iterExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    llvm::DenseSet<std::uint32_t> mappings;
    for (auto iter : mappingExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        mappings.insert(iter.getZExtValue());
    }
    int i = 0;
    llvm::interleaveComma(operands, printer,
                          [&](mlir::Value value)
                          {
                              if (iters.contains(i))
                              {
                                  printer << '*' << value;
                              }
                              else if (mappings.contains(i))
                              {
                                  printer << "**" << value;
                              }
                              else if (!keywords[i].cast<mlir::StringAttr>().getValue().empty())
                              {
                                  printer << keywords[i].cast<mlir::StringAttr>().getValue() << " = " << value;
                              }
                              else
                              {
                                  printer << value;
                              }
                              i++;
                          });
    printer << ')';
}

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

mlir::Attribute toBool(mlir::Attribute value)
{
    return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(value).Default({});
}

} // namespace


mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return constant();
}

mlir::OpFoldResult pylir::Py::GetAttrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (!operands[0])
    {
        return nullptr;
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(operands[0]).Default(mlir::OpFoldResult{nullptr});
}

mlir::OpFoldResult pylir::Py::GetItemOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto object = operands[0];
    auto index = operands[1];
    if (!object || !index)
    {
        return nullptr;
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(operands[0])
        .Case(
            [&](mlir::StringAttr attr) -> mlir::OpFoldResult
            {
                auto integer = index.dyn_cast_or_null<Py::IntAttr>();
                if (!integer)
                {
                    return nullptr;
                }
                auto utf32 = Text::toUTF32String(attr.getValue());
                auto value = integer.getValue();
                if (value.isNegative())
                {
                    value += BigInt(utf32.size());
                }
                if (value.isNegative() || value >= BigInt(utf32.size()))
                {
                    return nullptr;
                }
                auto codepoint = utf32[value.getInteger<std::size_t>()];
                auto utf8 = Text::toUTF8String({&codepoint, 1});
                return mlir::StringAttr::get(getContext(), utf8);
            })
        .Case<Py::TupleAttr, Py::ListAttr>(
            [&](auto sequence) -> mlir::OpFoldResult
            {
                auto array = sequence.getValue();
                auto integer = index.dyn_cast_or_null<Py::IntAttr>();
                if (!integer)
                {
                    return nullptr;
                }
                auto value = integer.getValue();
                if (value.isNegative())
                {
                    value += BigInt(array.size());
                }
                if (value.isNegative() || value >= BigInt(array.size()))
                {
                    return nullptr;
                }
                return array[value.getInteger<std::size_t>()];
            })
        .Case(
            [&](Py::DictAttr dictAttr) -> mlir::OpFoldResult
            {
                auto result = std::find_if(dictAttr.getValue().begin(), dictAttr.getValue().end(),
                                           [&](auto&& pair) { return pair.first == index; });
                if (result != dictAttr.getValue().end())
                {
                    return result->second;
                }
                return nullptr;
            })
        .Default(mlir::OpFoldResult{nullptr});
}

mlir::CallInterfaceCallable pylir::Py::CallOp::getCallableForCallee()
{
    return callee();
}

mlir::Operation::operand_range pylir::Py::CallOp::getArgOperands()
{
    return arguments();
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

mlir::LogicalResult pylir::Py::CallOp::inferReturnTypes(::mlir::MLIRContext* context,
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

mlir::LogicalResult pylir::Py::GetGlobalOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable)
{
    return mlir::success(symbolTable.lookupNearestSymbolFrom(*this, name()));
}

// TODO remove MLIR 14
using namespace mlir;

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>
