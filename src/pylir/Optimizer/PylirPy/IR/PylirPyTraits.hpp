#pragma once

#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/RuntimeTypeInterface.hpp>

namespace pylir::Py
{
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

#define BUILTIN(x, ...)                                                                                           \
    template <class ConcreteType>                                                                                 \
    class x##RuntimeType : public RuntimeTypeInterface::Trait<ConcreteType>                                       \
    {                                                                                                             \
    public:                                                                                                       \
        mlir::OpFoldResult getRuntimeType(unsigned)                                                               \
        {                                                                                                         \
            return mlir::FlatSymbolRefAttr::get(this->getOperation()->getContext(), pylir::Py::Builtins::x.name); \
        }                                                                                                         \
    };

#include <pylir/Interfaces/Builtins.def>

} // namespace pylir::Py
