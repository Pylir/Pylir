#pragma once

#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

#include "PylirPyTypes.hpp"

namespace pylir::Py
{

class LandingPadOp;

template <class ConcreteType>
class AlwaysBound : public mlir::OpTrait::TraitBase<ConcreteType, AlwaysBound>
{
    static mlir::LogicalResult verifyTrait(mlir::Operation*)
    {
        static_assert(!ConcreteType::template hasTrait<mlir::OpTrait::ZeroOperands>(),
                      "'Always Bound' trait is ony applicable to ops with results");
        return mlir::success();
    }
};

template <class ConcreteType>
class NoCapture : public CaptureInterface::Trait<ConcreteType>
{
public:
    bool capturesOperand(unsigned int)
    {
        return false;
    }
};

#define BUILTIN(x, ...)                                                                                               \
    template <class ConcreteType>                                                                                     \
    class x##RefinedType : public TypeRefineableInterface::Trait<ConcreteType>                                        \
    {                                                                                                                 \
    public:                                                                                                           \
        llvm::SmallVector<pylir::Py::ObjectTypeInterface> refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, \
                                                                      mlir::SymbolTable&)                             \
        {                                                                                                             \
            auto* context = this->getOperation()->getContext();                                                       \
            return {pylir::Py::ClassType::get(                                                                        \
                context, mlir::FlatSymbolRefAttr::get(context, pylir::Py::Builtins::x.name), llvm::None)};            \
        }                                                                                                             \
    };

#include <pylir/Interfaces/Builtins.def>

} // namespace pylir::Py
