
#include "X86_64.hpp"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Support/Variant.hpp>

namespace
{
llvm::SmallVector<mlir::Type> flatten(mlir::Type type)
{
    llvm::SmallVector<mlir::Type> result;
    if (auto structType = type.dyn_cast<mlir::LLVM::LLVMStructType>())
    {
        for (auto iter : structType.getBody())
        {
            auto temp = flatten(iter);
            result.insert(result.end(), temp.begin(), temp.end());
        }
    }
    else if (auto arrayType = type.dyn_cast<mlir::LLVM::LLVMArrayType>())
    {
        auto temp = flatten(arrayType.getElementType());
        for (std::size_t i = 0; i < arrayType.getNumElements(); i++)
        {
            result.insert(result.end(), temp.begin(), temp.end());
        }
    }
    else
    {
        result = {type};
    }
    return result;
}

} // namespace

std::tuple<pylir::X86_64::Adjustments::Arg, mlir::Type, mlir::Type>
    pylir::X86_64::flattenSingleArg(mlir::Type type, std::uint8_t* takenIntegers, std::uint8_t* takenFloats)
{
    constexpr std::uint8_t availableIntegerRegisters = 6;
    constexpr std::uint8_t availableFloatingPointRegisters = 8;
    Adjustments::Arg dest;
    std::size_t retIndex = 0;
    std::array<mlir::Type, 2> ret = {};

    std::uint8_t takenIntegerRegisters = takenIntegers ? *takenIntegers : 0;
    std::uint8_t takenFloatingPointRegisters = takenFloats ? *takenFloats : 0;
    const auto flat = flatten(type);
    auto iter = flat.begin();
    while (iter != flat.end())
    {
        const auto begin = iter;
        bool encounteredInteger = false;
        std::size_t size = 0;
        std::size_t currentAlignment = 0;
        while (size < 8 && iter != flat.end())
        {
            auto alignment = getSizeOf(*iter);
            const auto temp = llvm::alignTo(size, alignment);
            if (temp >= 8)
            {
                break;
            }
            size = temp;
            if (iter->isa<mlir::IntegerType, mlir::LLVM::LLVMPointerType>())
            {
                encounteredInteger = true;
            }
            // Special case X86_FP80Ty if ever used here
            currentAlignment = std::max(currentAlignment, alignment);
            size += alignment;
            iter++;
        }
        size = llvm::alignTo(size, currentAlignment);
        if (encounteredInteger)
        {
            // We encountered at least one integer therefore even if a floating point type was in there
            // it's gotta go into a integer register
            if (takenIntegerRegisters >= availableIntegerRegisters)
            {
                if (type.isa<mlir::LLVM::LLVMStructType>())
                {
                    return {OnStack{}, type, nullptr};
                }
                return {Unchanged{}, type, nullptr};
            }

            takenIntegerRegisters++;
            if (type.isa<mlir::LLVM::LLVMStructType>() && !std::holds_alternative<MultipleArgs>(dest))
            {
                dest = MultipleArgs{};
            }
            if (type.isa<mlir::LLVM::LLVMStructType>())
            {
                ret[retIndex++] = mlir::IntegerType::get(type.getContext(), size * 8);
            }
            else
            {
                ret[retIndex++] = type;
            }
            if (auto* multiArgs = std::get_if<MultipleArgs>(&dest))
            {
                multiArgs->size++;
            }
            continue;
        }
        if (std::distance(begin, iter) == 2 && begin->isa<mlir::Float32Type>() && (begin + 1)->isa<mlir::Float32Type>())
        {
            // Two floats can be packed as a single 64 bit value int oa  xmm register. This is represented as a vector
            // in LLVM IR
            if (takenFloatingPointRegisters >= availableFloatingPointRegisters)
            {
                if (type.isa<mlir::LLVM::LLVMStructType>())
                {
                    return {OnStack{}, type, nullptr};
                }
                return {Unchanged{}, type, nullptr};
            }
            takenFloatingPointRegisters++;
            if (!std::holds_alternative<MultipleArgs>(dest))
            {
                dest = MultipleArgs{};
            }
            ret[retIndex++] = mlir::LLVM::LLVMFixedVectorType::get(*begin, 2);
            pylir::get<MultipleArgs>(dest).size++;
            continue;
        }

        PYLIR_ASSERT(std::distance(begin, iter) == 1);
        // Must be a floating point type because if it were integer it would have taken the encounteredInteger branch
        // above
        if (takenFloatingPointRegisters >= availableFloatingPointRegisters)
        {
            if (type.isa<mlir::LLVM::LLVMStructType>())
            {
                return {OnStack{}, type, nullptr};
            }
            return {Unchanged{}, type, nullptr};
        }
        takenFloatingPointRegisters++;
        if (type.isa<mlir::LLVM::LLVMStructType>() && !std::holds_alternative<MultipleArgs>(dest))
        {
            dest = MultipleArgs{};
        }
        ret[retIndex++] = *begin;
        if (auto* multiArgs = std::get_if<MultipleArgs>(&dest))
        {
            multiArgs->size++;
        }
    }
    if (takenFloats)
    {
        *takenFloats = takenFloatingPointRegisters;
    }
    if (takenIntegers)
    {
        *takenIntegers = takenIntegerRegisters;
    }
    return {dest, ret[0], ret[1]};
}

mlir::LLVM::LLVMFuncOp pylir::X86_64::declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
                                                       mlir::Type returnType, llvm::StringRef name,
                                                       mlir::TypeRange inputTypes)
{
    Adjustments adjustments;
    adjustments.arguments.reserve(inputTypes.size());
    adjustments.originalRetType = returnType;

    mlir::Type retType = returnType;
    llvm::SmallVector<mlir::Type> argumentTypes;
    if (!returnType.isa<mlir::LLVM::LLVMVoidType>())
    {
        auto size = getSizeOf(returnType);
        if (size > 16)
        {
            adjustments.returnType = PointerToTemporary{};
            argumentTypes.push_back(mlir::LLVM::LLVMPointerType::get(returnType));
            retType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
        }
        else
        {
            bool wasStruct = returnType.isa<mlir::LLVM::LLVMStructType>();
            std::pair<mlir::Type, mlir::Type> types;
            std::tie(std::ignore, types.first, types.second) = flattenSingleArg(returnType);
            if (types.second)
            {
                retType = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), {types.first, types.second});
            }
            else
            {
                retType = types.first;
            }
            if (wasStruct)
            {
                adjustments.returnType = Flattened{};
            }
        }
    }

    std::uint8_t takenIntegerRegisters = 0;
    std::uint8_t takenFloatingPointRegisters = 0;
    for (auto inputType : inputTypes)
    {
        if (getSizeOf(inputType) > 16)
        {
            adjustments.arguments.emplace_back(OnStack{});
            argumentTypes.push_back(mlir::LLVM::LLVMPointerType::get(inputType));
            continue;
        }
        std::pair<mlir::Type, mlir::Type> types;
        Adjustments::Arg dest;
        std::tie(dest, types.first, types.second) =
            flattenSingleArg(inputType, &takenIntegerRegisters, &takenFloatingPointRegisters);
        adjustments.arguments.push_back(dest);
        if (std::holds_alternative<OnStack>(dest))
        {
            argumentTypes.push_back(mlir::LLVM::LLVMPointerType::get(inputType));
        }
        else if (std::holds_alternative<MultipleArgs>(dest))
        {
            argumentTypes.push_back(types.first);
            if (types.second)
            {
                argumentTypes.push_back(types.second);
            }
        }
        else
        {
            argumentTypes.push_back(inputType);
        }
    }

    auto funcOp =
        builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, mlir::LLVM::LLVMFunctionType::get(retType, argumentTypes));
    std::size_t argStart = 0;
    if (std::holds_alternative<PointerToTemporary>(adjustments.returnType))
    {
        argStart = 1;
        funcOp.setArgAttrs(0,
                           {builder.getNamedAttr(mlir::LLVM::LLVMDialect::getNoAliasAttrName(), builder.getUnitAttr()),
                            builder.getNamedAttr("llvm.sret", builder.getUnitAttr())});
    }
    std::size_t origArgI = 0;
    for (std::size_t i = argStart; i < funcOp.getNumArguments(); origArgI++)
    {
        pylir::match(
            adjustments.arguments[origArgI], [&](Unchanged) { i++; },
            [&](OnStack)
            {
                auto exit = llvm::make_scope_exit([&] { i++; });
                funcOp.setArgAttrs(i, {builder.getNamedAttr("llvm.byval", builder.getUnitAttr()),
                                       builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                                                            builder.getI64IntegerAttr(8))});
            },
            [&](MultipleArgs multipleArgs) { i += multipleArgs.size; });
    }
    m_adjustments.insert({funcOp, adjustments});
    return funcOp;
}

mlir::Value pylir::X86_64::callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                                         mlir::ValueRange operands)
{
    auto result = m_adjustments.find(func);
    PYLIR_ASSERT(result != m_adjustments.end());
    auto& adjustments = result->second;
    mlir::LLVM::AllocaOp returnSlot;

    llvm::SmallVector<mlir::Value> arguments;

    if (std::holds_alternative<PointerToTemporary>(adjustments.returnType))
    {
        auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        returnSlot =
            builder.create<mlir::LLVM::AllocaOp>(loc, func.getType().getParams().front(), one, mlir::IntegerAttr{});
        arguments.push_back(returnSlot);
    }

    for (auto iter = operands.begin(); iter != operands.end(); iter++)
    {
        auto getPointerToMemory = [&](mlir::Value value) -> mlir::Value
        {
            if (auto load = value.getDefiningOp<mlir::LLVM::LoadOp>())
            {
                return load.getAddr();
            }
            auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
            auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
                loc, mlir::LLVM::LLVMPointerType::get((*iter).getType()), one, mlir::IntegerAttr{});
            builder.create<mlir::LLVM::StoreOp>(loc, *iter, tempAlloca);
            return tempAlloca;
        };
        pylir::match(
            adjustments.arguments[iter - operands.begin()], [&](Unchanged) { arguments.push_back(*iter); },
            [&](OnStack) { arguments.push_back(getPointerToMemory(*iter)); },
            [&](MultipleArgs multipleArgs)
            {
                auto address = getPointerToMemory(*iter);
                if (multipleArgs.size == 1)
                {
                    auto paramType = func.getType().getParamType(arguments.size());
                    auto casted = builder.create<mlir::LLVM::BitcastOp>(
                        loc, mlir::LLVM::LLVMPointerType::get(paramType), address);
                    arguments.push_back(builder.create<mlir::LLVM::LoadOp>(loc, casted));
                    return;
                }
                auto firstType = func.getType().getParamType(arguments.size());
                auto secondType = func.getType().getParamType(arguments.size() + 1);
                auto zero =
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
                auto one =
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
                auto firstValue = builder.create<mlir::LLVM::GEPOp>(loc, mlir::LLVM::LLVMPointerType::get(firstType),
                                                                    address, mlir::ValueRange{zero, zero});
                auto secondValue = builder.create<mlir::LLVM::GEPOp>(loc, mlir::LLVM::LLVMPointerType::get(secondType),
                                                                     address, mlir::ValueRange{zero, one});
                arguments.push_back(builder.create<mlir::LLVM::LoadOp>(loc, firstValue));
                arguments.push_back(builder.create<mlir::LLVM::LoadOp>(loc, secondValue));
            });
    }

    auto call = builder.create<mlir::LLVM::CallOp>(loc, func, arguments);
    return pylir::match(
        adjustments.returnType,
        [&](Unchanged) -> mlir::Value
        {
            if (call->getNumResults() > 0)
            {
                return call.getResult(0);
            }
            return {};
        },
        [&](PointerToTemporary) -> mlir::Value { return builder.create<mlir::LLVM::LoadOp>(loc, returnSlot); },
        [&](Flattened) -> mlir::Value
        {
            auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
            auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
                loc, mlir::LLVM::LLVMPointerType::get(adjustments.originalRetType), one, mlir::IntegerAttr{});
            auto casted = builder.create<mlir::LLVM::BitcastOp>(
                loc, mlir::LLVM::LLVMPointerType::get(call.getResult(0).getType()), tempAlloca);
            builder.create<mlir::LLVM::StoreOp>(loc, call.getResult(0), casted);
            return builder.create<mlir::LLVM::LoadOp>(loc, tempAlloca);
        });
}

pylir::X86_64::X86_64(mlir::DataLayout dataLayout) : CABI(std::move(dataLayout)) {}
