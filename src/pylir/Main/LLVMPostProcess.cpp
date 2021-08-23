
#include "LLVMPostProcess.hpp"

#include <llvm/IR/IRBuilder.h>

#include <pylir/Optimizer/Dialect/PylirTypeObjects.hpp>

void pylir::postProcessLLVMModule(llvm::Module& module)
{
    // __main__.__init__ in the future
    if (auto* aliasee = module.getFunction("__init__"))
    {
        // Normal name so it can be called from C
        llvm::GlobalAlias::create("pylir__main____init__", aliasee);
    }

    if (auto* intTypeObject = module.getGlobalVariable(pylir::Dialect::intTypeObjectName))
    {
        llvm::GlobalAlias::create(llvm::GlobalValue::ExternalLinkage, "pylir_integer_type_object", intTypeObject);
    }

    // Add allocsize that can't be set in MLIR LLVM IR
    if (auto alloc = module.getFunction("pylir_gc_alloc"))
    {
        alloc->addFnAttr(llvm::Attribute::getWithAllocSizeArgs(alloc->getContext(), 0, {}));
    }

    // Apply comdat to all globals with linkonce_odr
    for (auto& iter : module.global_objects())
    {
        if (iter.getLinkage() == llvm::GlobalValue::LinkOnceODRLinkage)
        {
            auto* comdat = module.getOrInsertComdat(iter.getName());
            comdat->setSelectionKind(llvm::Comdat::Any);
            iter.setComdat(comdat);
        }
    }

    // Apply comdat to all functions in the ctor array
    auto* ctors = module.getGlobalVariable("llvm.global_ctors");
    if (!ctors)
    {
        return;
    }
    auto array = llvm::dyn_cast<llvm::ConstantArray>(ctors->getInitializer());
    if (!array)
    {
        return;
    }
    llvm::Constant* element;
    for (std::size_t i = 0; (element = array->getAggregateElement(i)); i++)
    {
        if (!llvm::isa<llvm::ConstantStruct>(element))
        {
            continue;
        }
        auto* function = llvm::dyn_cast_or_null<llvm::Function>(element->getAggregateElement(1));
        if (!function)
        {
            continue;
        }
        auto* comdat = module.getOrInsertComdat(function->getName());
        comdat->setSelectionKind(llvm::Comdat::Any);
        function->setComdat(comdat);
    }
}
