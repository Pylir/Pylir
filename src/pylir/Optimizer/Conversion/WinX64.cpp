
#include "WinX64.hpp"

#include <mlir/IR/Builders.h>

#include <pylir/Support/Macros.hpp>

namespace
{
bool isLegalIntegerSize(std::size_t size)
{
    return size == 8 || size == 16 || size == 32 || size == 64;
}
} // namespace

mlir::LLVM::LLVMFuncOp pylir::Dialect::WinX64::declareFunc(mlir::OpBuilder& builder, mlir::Location loc,
                                                           mlir::Type returnType, llvm::StringRef name,
                                                           mlir::TypeRange inputTypes)
{
    mlir::Type retType = returnType;
    llvm::SmallVector<mlir::Type> argumentTypes;
    argumentTypes.reserve(inputTypes.size());

    Adjustments adjustments;
    adjustments.originalRetType = returnType;
    adjustments.arguments.resize(inputTypes.size());

    if (returnType.isa<mlir::LLVM::LLVMStructType>())
    {
        auto size = getSizeOf(returnType);
        if (isLegalIntegerSize(size))
        {
            adjustments.returnType = IntegerRegister;
            retType = builder.getIntegerType(size * 8);
        }
        else
        {
            adjustments.returnType = PointerToTemporary;
            retType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
            argumentTypes.push_back(mlir::LLVM::LLVMPointerType::get(returnType));
        }
    }

    for (std::size_t i = 0; i < inputTypes.size(); i++)
    {
        if (!inputTypes[i].isa<mlir::LLVM::LLVMStructType>())
        {
            argumentTypes.emplace_back(inputTypes[i]);
            continue;
        }
        auto size = getSizeOf(inputTypes[i]);
        if (isLegalIntegerSize(size))
        {
            argumentTypes.push_back(builder.getIntegerType(size * 8));
            adjustments.arguments[i] = IntegerRegister;
        }
        else
        {
            argumentTypes.push_back(mlir::LLVM::LLVMPointerType::get(inputTypes[i]));
            adjustments.arguments[i] = PointerToTemporary;
        }
    }

    auto funcOp =
        builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, mlir::LLVM::LLVMFunctionType::get(retType, argumentTypes));
    if (adjustments.returnType == PointerToTemporary)
    {
        funcOp.setArgAttrs(0,
                           {builder.getNamedAttr(mlir::LLVM::LLVMDialect::getNoAliasAttrName(), builder.getUnitAttr()),
                            builder.getNamedAttr("llvm.sret", builder.getUnitAttr())});
    }
    m_adjustments.insert({funcOp, adjustments});
    return funcOp;
}

mlir::Value pylir::Dialect::WinX64::callFunc(mlir::OpBuilder& builder, mlir::Location loc, mlir::LLVM::LLVMFuncOp func,
                                             mlir::ValueRange operands)
{
    auto result = m_adjustments.find(func);
    PYLIR_ASSERT(result != m_adjustments.end());
    auto& adjustments = result->second;
    mlir::LLVM::AllocaOp returnSlot;

    llvm::SmallVector<mlir::Value> arguments;

    std::size_t paramBegin = 0;
    if (adjustments.returnType == PointerToTemporary)
    {
        paramBegin = 1;
        auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        returnSlot =
            builder.create<mlir::LLVM::AllocaOp>(loc, func.getType().getParams().front(), one, mlir::IntegerAttr{});
        arguments.push_back(returnSlot);
    }

    for (std::size_t i = 0; i < operands.size(); i++)
    {
        switch (adjustments.arguments[i])
        {
            case Nothing: arguments.push_back(operands[i]); break;
            case IntegerRegister:
            {
                auto integerPointerType = mlir::LLVM::LLVMPointerType::get(func.getType().getParams()[paramBegin + i]);
                if (auto load = operands[i].getDefiningOp<mlir::LLVM::LoadOp>())
                {
                    auto casted = builder.create<mlir::LLVM::BitcastOp>(loc, integerPointerType, load.addr());
                    arguments.push_back(builder.create<mlir::LLVM::LoadOp>(loc, casted));
                    break;
                }
                auto one =
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
                auto tempAlloca =
                    builder.create<mlir::LLVM::AllocaOp>(loc, integerPointerType, one, mlir::IntegerAttr{});
                auto casted = builder.create<mlir::LLVM::BitcastOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(operands[i].getType()), tempAlloca);
                builder.create<mlir::LLVM::StoreOp>(loc, operands[i], casted);
                arguments.push_back(builder.create<mlir::LLVM::LoadOp>(loc, tempAlloca));
                break;
            }
            case PointerToTemporary:
            {
                auto one =
                    builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
                auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
                    loc, mlir::LLVM::LLVMPointerType::get(operands[i].getType()), one, mlir::IntegerAttr{});
                arguments.push_back(tempAlloca);
                if (auto load = operands[i].getDefiningOp<mlir::LLVM::LoadOp>())
                {
                    auto falseConstant =
                        builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                    auto null = builder.create<mlir::LLVM::NullOp>(loc, tempAlloca.getType());
                    auto gep = builder.create<mlir::LLVM::GEPOp>(loc, null.getType(), null, mlir::ValueRange{one});
                    auto size = builder.create<mlir::LLVM::PtrToIntOp>(loc, builder.getI64Type(), gep);
                    builder.create<mlir::LLVM::MemcpyOp>(loc, tempAlloca, load.addr(), size, falseConstant);
                    break;
                }

                builder.create<mlir::LLVM::StoreOp>(loc, operands[i], tempAlloca);
                break;
            }
        }
    }

    auto call = builder.create<mlir::LLVM::CallOp>(loc, func, arguments);

    switch (adjustments.returnType)
    {
        case Nothing:
            if (call.getNumResults() == 0)
            {
                return {};
            }
            return call.getResult(0);
        case IntegerRegister:
        {
            auto one = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
            auto tempAlloca = builder.create<mlir::LLVM::AllocaOp>(
                loc, mlir::LLVM::LLVMPointerType::get(adjustments.originalRetType), one, mlir::IntegerAttr{});
            auto casted = builder.create<mlir::LLVM::BitcastOp>(
                loc, mlir::LLVM::LLVMPointerType::get(call.getResult(0).getType()), tempAlloca);
            builder.create<mlir::LLVM::StoreOp>(loc, call.getResult(0), casted);
            return builder.create<mlir::LLVM::LoadOp>(loc, tempAlloca);
        }
        case PointerToTemporary: return builder.create<mlir::LLVM::LoadOp>(loc, returnSlot);
    }
}

pylir::Dialect::WinX64::WinX64(mlir::DataLayout dataLayout) : CABI(std::move(dataLayout)) {}
