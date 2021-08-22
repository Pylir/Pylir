#include "CABI.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

std::size_t pylir::Dialect::CABI::getSizeOf(mlir::Type type) const
{
    // For now assumes sizeof == alignof for all primitives
    return llvm::TypeSwitch<mlir::Type, std::size_t>(type)
        .Case<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMPointerType>(
            [this](mlir::Type type) { return m_dataLayout.getTypeSize(type); })
        .Case<mlir::LLVM::LLVMArrayType>([this](mlir::LLVM::LLVMArrayType array)
                                         { return getSizeOf(array.getElementType()) * array.getNumElements(); })
        .Case<mlir::LLVM::LLVMStructType>(
            [this](mlir::LLVM::LLVMStructType structType)
            {
                PYLIR_ASSERT((!structType.isIdentified() || structType.isInitialized()) && !structType.isPacked());
                std::size_t size = 0;
                std::size_t alignment = 0;
                for (auto iter : structType.getBody())
                {
                    auto elementSize = getSizeOf(iter);
                    alignment = std::max(elementSize, alignment);
                    size = llvm::alignTo(size, elementSize);
                    size += elementSize;
                }
                size = llvm::alignTo(size, alignment);
                return size;
            })
        .Default([](auto) -> std::size_t { PYLIR_UNREACHABLE; });
}
