#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>

#include <pylir/Support/Macros.hpp>

#include <memory>

namespace pylir
{

class MemoryAccess
{
public:
    enum MemoryAccessKind
    {
        MemoryDef,
        MemoryPhi,
        MemoryUse,
    };

private:
    const MemoryAccessKind m_kind;

protected:
    explicit MemoryAccess(MemoryAccessKind kind) : m_kind(kind) {}

public:
    virtual ~MemoryAccess() = default;

    MemoryAccessKind getKind() const
    {
        return m_kind;
    }
};

class MemoryDef final : public MemoryAccess
{
    mlir::Operation* m_memoryInstruction;
    MemoryAccess* PYLIR_NULLABLE m_previous;

public:
    MemoryDef(mlir::Operation* memoryInstruction, MemoryAccess* PYLIR_NULLABLE previous)
        : MemoryAccess(MemoryAccess::MemoryDef), m_memoryInstruction(memoryInstruction), m_previous(previous)
    {
    }

    static bool classof(const MemoryAccess* memoryAccess)
    {
        return memoryAccess->getKind() == MemoryAccess::MemoryDef;
    }

    mlir::Operation* getMemoryInstruction() const
    {
        return m_memoryInstruction;
    }

    MemoryAccess* getPrevious() const
    {
        return m_previous;
    }
};

class MemoryUse final : public MemoryAccess
{
    mlir::Operation* m_memoryInstruction;
    MemoryAccess* PYLIR_NULLABLE m_definition;
    mlir::AliasResult m_aliasResult;

public:
    MemoryUse(mlir::Operation* memoryInstruction, MemoryAccess* PYLIR_NULLABLE definition,
              mlir::AliasResult aliasResult)
        : MemoryAccess(MemoryAccess::MemoryUse),
          m_memoryInstruction(memoryInstruction),
          m_definition(definition),
          m_aliasResult(aliasResult)
    {
    }

    static bool classof(const MemoryAccess* memoryAccess)
    {
        return memoryAccess->getKind() == MemoryAccess::MemoryUse;
    }

    mlir::Operation* getMemoryInstruction() const
    {
        return m_memoryInstruction;
    }

    MemoryAccess* getDefinition() const
    {
        return m_definition;
    }

    const mlir::AliasResult& getAliasResult() const
    {
        return m_aliasResult;
    }
};

class MemoryPhi final : public MemoryAccess
{
    mlir::Block* m_block;
    std::vector<std::pair<mlir::Block*, MemoryAccess*>> m_incoming;

public:
    MemoryPhi(mlir::Block* block, std::vector<std::pair<mlir::Block*, MemoryAccess*>> incoming)
        : MemoryAccess(MemoryAccess::MemoryPhi), m_block(block), m_incoming(std::move(incoming))
    {
    }

    static bool classof(const MemoryAccess* memoryAccess)
    {
        return memoryAccess->getKind() == MemoryAccess::MemoryPhi;
    }

    mlir::Block* getBlock() const
    {
        return m_block;
    }

    llvm::ArrayRef<std::pair<mlir::Block*, MemoryAccess*>> getIncoming() const
    {
        return m_incoming;
    }
};

class MemorySSA
{
    llvm::DenseMap<mlir::PointerUnion<mlir::Operation*, mlir::Block*>, std::unique_ptr<MemoryAccess>> m_results;

public:
    explicit MemorySSA(mlir::Operation* operation, mlir::AnalysisManager& analysisManager);

    MemoryAccess* getMemoryAccess(mlir::Operation* operation);

    MemoryPhi* getMemoryAccess(mlir::Block* block);
};
} // namespace pylir
