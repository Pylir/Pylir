#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/RegionGraphTraits.h>

#include <llvm/ADT/PostOrderIterator.h>

#include <pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>
#include <pylir/Support/Variant.hpp>

#include <optional>
#include <queue>
#include <stack>
#include <variant>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
class Monomorph : public pylir::Py::MonomorphBase<Monomorph>
{
protected:
    void runOnOperation() override;
};

struct FunctionSpecialization
{
    mlir::Operation* function;
    std::vector<pylir::Py::ObjectTypeInterface> argTypes;
};
} // namespace

template <>
struct llvm::DenseMapInfo<FunctionSpecialization>
{
    static inline FunctionSpecialization getEmptyKey()
    {
        return {
            llvm::DenseMapInfo<mlir::Operation*>::getEmptyKey(),
            {},
        };
    }

    static inline FunctionSpecialization getTombstoneKey()
    {
        return {
            llvm::DenseMapInfo<mlir::Operation*>::getTombstoneKey(),
            {},
        };
    }

    static inline unsigned getHashValue(const FunctionSpecialization& value)
    {
        return llvm::hash_combine(value.function,
                                  llvm::hash_combine_range(value.argTypes.begin(), value.argTypes.end()));
    }

    static inline bool isEqual(const FunctionSpecialization& lhs, const FunctionSpecialization& rhs)
    {
        return std::tie(lhs.function, lhs.argTypes) == std::tie(rhs.function, rhs.argTypes);
    }
};

namespace
{

// TODO: Figure out a supremum to guarantee termination.
struct Lattice
{
    pylir::Py::ObjectTypeInterface type;

    bool merge(const Lattice& rhs)
    {
        auto newType = pylir::Py::joinTypes(type, rhs.type);
        bool changed = newType != type;
        type = newType;
        return changed;
    }
};

struct FinishedAnalysis
{
    pylir::Py::ObjectTypeInterface returnType;
    llvm::DenseMap<mlir::Value, Lattice> lattices;
};

class MonomorphModuleInfo
{
    llvm::MapVector<FunctionSpecialization, FinishedAnalysis> m_calculatedSpecializations;

public:
    [[nodiscard]] const FinishedAnalysis* lookup(const FunctionSpecialization& specialization) const
    {
        auto result = m_calculatedSpecializations.find(specialization);
        if (result == m_calculatedSpecializations.end())
        {
            return nullptr;
        }
        return &result->second;
    }

    auto insert(FunctionSpecialization&& specialization, FinishedAnalysis finishedAnalysis)
    {
        return m_calculatedSpecializations.insert({std::move(specialization), std::move(finishedAnalysis)});
    }

    [[nodiscard]] const llvm::MapVector<FunctionSpecialization, FinishedAnalysis>& getCalculatedSpecializations() const
    {
        return m_calculatedSpecializations;
    }
};

enum class RunStage
{
    CallAnswer,
    Initial
};

class MonomorphFunctionImpl
{
    mlir::SymbolTable* m_symbolTable;
    mlir::FunctionOpInterface m_functionOp;
    std::vector<pylir::Py::ObjectTypeInterface> m_argTypes;
    std::queue<mlir::Block*> m_workList;
    llvm::DenseSet<mlir::Block*> m_inWorkList;
    llvm::DenseMap<mlir::Value, Lattice> m_lattices;
    mlir::Block::iterator m_currentOp;
    bool m_inBlockChanged = false;
    pylir::Py::ObjectTypeInterface m_returnType;

    void addLattice(mlir::Value value, const Lattice& lattice)
    {
        auto result = m_lattices.find(value);
        if (result != m_lattices.end())
        {
            if (!result->second.merge(lattice))
            {
                return;
            }
        }
        else
        {
            m_lattices.insert({value, lattice});
        }
        m_inBlockChanged = true;
    }

    std::vector<FunctionSpecialization> evalCallOp(mlir::CallOpInterface callOp)
    {
        mlir::FunctionOpInterface functionOpInterface;
        auto callee = callOp.getCallableForCallee();
        std::vector<pylir::Py::ObjectTypeInterface> argTypes;
        for (auto iter : callOp.getArgOperands())
        {
            auto lattice = m_lattices.find(iter);
            if (lattice != m_lattices.end())
            {
                argTypes.push_back(lattice->second.type);
                continue;
            }
            if (auto objType = iter.getType().dyn_cast<pylir::Py::ObjectTypeInterface>())
            {
                argTypes.push_back(objType);
            }
        }
        if (auto ref = callee.dyn_cast<mlir::SymbolRefAttr>())
        {
            functionOpInterface = m_symbolTable->lookup<mlir::FunctionOpInterface>(ref.getLeafReference());
            return {FunctionSpecialization{functionOpInterface, std::move(argTypes)}};
        }

        auto pessimize = [&]
        {
            for (auto iter : callOp->getResults())
            {
                addLattice(iter, Lattice{pylir::Py::UnknownType::get(m_functionOp.getContext())});
            }
        };

        auto mroLookup = callee.get<mlir::Value>().getDefiningOp<pylir::Py::MROLookupOp>();
        if (!mroLookup)
        {
            pessimize();
            return {};
        }
        auto mroTuple = mroLookup.getMroTuple().getDefiningOp<pylir::Py::TypeMROOp>();
        if (!mroTuple)
        {
            pessimize();
            return {};
        }
        auto typeOf = mroTuple.getTypeObject().getDefiningOp<pylir::Py::TypeOfOp>();
        if (!typeOf)
        {
            pessimize();
            return {};
        }
        auto result = m_lattices.find(typeOf.getObject());
        if (result == m_lattices.end())
        {
            pessimize();
            return {};
        }
        std::vector<FunctionSpecialization> results;
        auto calcFunc = [&](mlir::FlatSymbolRefAttr type) -> mlir::FunctionOpInterface
        {
            auto typeObject = m_symbolTable->lookup<pylir::Py::GlobalValueOp>(type.getAttr());
            if (typeObject.isDeclaration())
            {
                return {};
            }
            auto slots = typeObject.getInitializerAttr().getSlots();
            auto slot = slots.get(mroLookup.getSlotAttr()).dyn_cast_or_null<pylir::Py::FunctionAttr>();
            if (!slot)
            {
                return {};
            }
            return m_symbolTable->lookup<mlir::FunctionOpInterface>(slot.getValue().getAttr());
        };
        auto type = result->second.type;
        if (auto var = type.dyn_cast<pylir::Py::VariantType>())
        {
            for (auto& iter : var.getElements())
            {
                auto typeObject = iter.getTypeObject();
                if (!typeObject)
                {
                    pessimize();
                    return {};
                }
                auto func = calcFunc(typeObject);
                if (!func)
                {
                    pessimize();
                    return {};
                }
                results.push_back({func, argTypes});
            }
        }
        else if (auto typeObject = type.getTypeObject())
        {
            auto func = calcFunc(typeObject);
            if (!func)
            {
                pessimize();
                return {};
            }
            results = {{func, std::move(argTypes)}};
        }

        return results;
    }

    void evalBlockArgs(mlir::Block* block)
    {
        llvm::SmallVector<llvm::Optional<Lattice>> blockArgs(block->getNumArguments());
        for (auto pred = block->pred_begin(); pred != block->pred_end(); pred++)
        {
            auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>((*pred)->getTerminator());
            PYLIR_ASSERT(branchOp);
            auto ops = branchOp.getSuccessorOperands(pred.getSuccessorIndex());
            PYLIR_ASSERT(ops);
            for (auto [blockArg, value] : llvm::zip(blockArgs, *ops))
            {
                auto result = m_lattices.find(value);
                if (result == m_lattices.end())
                {
                    continue;
                }
                if (!blockArg)
                {
                    blockArg = result->second;
                    continue;
                }
                blockArg->merge(result->second);
            }
        }
        for (auto [blockArg, lattice] : llvm::zip(block->getArguments(), blockArgs))
        {
            if (blockArg.getType().isa<pylir::Py::ObjectTypeInterface>())
            {
                PYLIR_ASSERT(lattice);
                addLattice(blockArg, *lattice);
            }
        }
    }

    void evalTerminator(mlir::Operation* terminator)
    {
        if (auto retOp = mlir::dyn_cast<pylir::Py::ReturnOp>(terminator); retOp && retOp.operands().size() == 1)
        {
            m_returnType = pylir::Py::joinTypes(m_returnType, m_lattices.find(retOp.operands()[0])->second.type);
        }
        if (!m_inBlockChanged)
        {
            return;
        }
        // TODO: handle `py.is` as condition to terminators
        for (auto* iter : terminator->getSuccessors())
        {
            if (m_inWorkList.insert(iter).second)
            {
                m_workList.push(iter);
            }
        }
    }

public:
    explicit MonomorphFunctionImpl(mlir::SymbolTable& symbolTable, mlir::FunctionOpInterface interface,
                                   std::vector<pylir::Py::ObjectTypeInterface>&& argTypes = {})
        : m_symbolTable(&symbolTable),
          m_functionOp(interface),
          m_argTypes(std::move(argTypes)),
          m_returnType(pylir::Py::UnboundType::get(interface->getContext()))
    {
        std::deque<mlir::Block*> rpo;
        std::copy(llvm::po_begin(&interface.getBody()), llvm::po_end(&interface.getBody()), std::front_inserter(rpo));
        std::for_each(rpo.begin() + 1, rpo.end(), [&](mlir::Block* block) { m_inWorkList.insert(block); });
        m_workList = std::queue{std::move(rpo)};
        m_currentOp = m_workList.front()->begin();
        PYLIR_ASSERT(m_argTypes.size() == interface.getNumArguments());
        for (auto [arg, type] : llvm::zip(interface.getArguments(), m_argTypes))
        {
            addLattice(arg, {type});
        }
    }

    ~MonomorphFunctionImpl() = default;
    MonomorphFunctionImpl(const MonomorphFunctionImpl&) = delete;
    MonomorphFunctionImpl& operator=(const MonomorphFunctionImpl&) = delete;
    MonomorphFunctionImpl(MonomorphFunctionImpl&&) noexcept = default;
    MonomorphFunctionImpl& operator=(MonomorphFunctionImpl&&) noexcept = default;

    using RunResponse = std::variant<pylir::Py::ObjectTypeInterface, std::vector<FunctionSpecialization>>;

    RunResponse run(RunStage stage, pylir::Py::ObjectTypeInterface returnType)
    {
        while (!m_workList.empty())
        {
            auto* currentBlock = m_workList.front();
            for (; m_currentOp != currentBlock->end(); m_currentOp++)
            {
                if (auto callable = mlir::dyn_cast<mlir::CallOpInterface>(m_currentOp))
                {
                    if (stage == RunStage::CallAnswer)
                    {
                        stage = RunStage::Initial;
                        PYLIR_ASSERT(callable->getNumResults() == 1);
                        PYLIR_ASSERT(returnType);
                        addLattice(callable->getResult(0), {returnType});
                    }
                    else if (auto specs = evalCallOp(callable); !specs.empty())
                    {
                        return {std::move(specs)};
                    }
                }

                if (m_currentOp->hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    evalTerminator(&(*m_currentOp));
                }

                auto typeRefinement = mlir::dyn_cast<pylir::Py::TypeRefineableInterface>(m_currentOp);
                if (!typeRefinement)
                {
                    for (auto iter : m_currentOp->getResults())
                    {
                        if (iter.getType().isa<pylir::Py::ObjectTypeInterface>())
                        {
                            addLattice(iter, {iter.getType()});
                        }
                    }
                    continue;
                }

                llvm::SmallVector<pylir::Py::ObjectTypeInterface> operandTypes(m_currentOp->getNumOperands());
                for (const auto& iter : llvm::enumerate(m_currentOp->getOperands()))
                {
                    if (iter.value().getType().isa<pylir::Py::ObjectTypeInterface>())
                    {
                        operandTypes[iter.index()] = m_lattices.find(iter.value())->second.type;
                    }
                }
                auto results = typeRefinement.refineTypes(operandTypes, *m_symbolTable);
                PYLIR_ASSERT(results.size() == m_currentOp->getNumResults());
                for (auto [res, type] : llvm::zip(m_currentOp->getResults(), results))
                {
                    addLattice(res, {type});
                }
            }
            m_workList.pop();
            m_inBlockChanged = false;
            if (!m_workList.empty())
            {
                evalBlockArgs(m_workList.front());
                m_inWorkList.erase(m_workList.front());
                m_currentOp = m_workList.front()->begin();
            }
        }
        return m_returnType;
    }

    [[nodiscard]] const mlir::FunctionOpInterface& getFunctionOp() const
    {
        return m_functionOp;
    }

    std::vector<pylir::Py::ObjectTypeInterface>&& getArgTypes() &&
    {
        return std::move(m_argTypes);
    }

    llvm::DenseMap<mlir::Value, Lattice>&& getLattices() &&
    {
        return std::move(m_lattices);
    }
};

class ExecutionFrame
{
    MonomorphFunctionImpl monomorphFunction;
    RunStage currentStage = RunStage::Initial;
    pylir::Py::ObjectTypeInterface returnType;
    std::size_t parentFrame;

public:
    explicit ExecutionFrame(MonomorphFunctionImpl&& monomorphFunction, std::size_t parentFrame = -1)
        : monomorphFunction(std::move(monomorphFunction)), parentFrame(parentFrame)
    {
    }

    MonomorphFunctionImpl::RunResponse run()
    {
        return monomorphFunction.run(std::exchange(currentStage, RunStage::CallAnswer), std::exchange(returnType, {}));
    }

    void addCallResult(pylir::Py::ObjectTypeInterface result)
    {
        if (!returnType)
        {
            returnType = result;
            return;
        }
        returnType = pylir::Py::joinTypes(returnType, result);
    }

    [[nodiscard]] std::size_t getParentFrame() const
    {
        return parentFrame;
    }

    MonomorphFunctionImpl&& getMonomorphFunction() &&
    {
        return std::move(monomorphFunction);
    }

    [[nodiscard]] const MonomorphFunctionImpl& getMonomorphFunction() const&
    {
        return monomorphFunction;
    }
};

void Monomorph::runOnOperation()
{
    mlir::SymbolTable symbolTable(getOperation());
    llvm::SmallVector<mlir::FunctionOpInterface> roots;
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
    {
        if (llvm::none_of(iter.getArgumentTypes(),
                          [](mlir::Type type) { return type.isa<pylir::Py::ObjectTypeInterface>(); }))
        {
            roots.push_back(iter);
        }
    }

    MonomorphModuleInfo info;
    for (auto& iter : roots)
    {
        llvm::DenseSet<FunctionSpecialization> currentlyExecuting;
        std::vector<ExecutionFrame> currentExecutionStack;
        currentExecutionStack.emplace_back(MonomorphFunctionImpl{symbolTable, iter});
        currentlyExecuting.insert({iter, {}});
        while (!currentExecutionStack.empty())
        {
            MonomorphFunctionImpl::RunResponse continuation;
            while (std::holds_alternative<std::vector<FunctionSpecialization>>(continuation =
                                                                                   currentExecutionStack.back().run()))
            {
                std::size_t parentFrame = currentExecutionStack.size() - 1;
                auto& funcSpecs = pylir::get<std::vector<FunctionSpecialization>>(continuation);
                for (auto& funcSpec : funcSpecs)
                {
                    if (const auto* result = info.lookup(funcSpec))
                    {
                        currentExecutionStack[parentFrame].addCallResult(result->returnType);
                        continue;
                    }
                    if (!currentlyExecuting.insert(funcSpec).second)
                    {
                        // Recursive case. TODO: Think about how to handle this. Placeholder type maybe?
                        currentExecutionStack[parentFrame].addCallResult(pylir::Py::UnknownType::get(&getContext()));
                        continue;
                    }
                    currentExecutionStack.emplace_back(
                        MonomorphFunctionImpl{symbolTable, mlir::cast<mlir::FunctionOpInterface>(funcSpec.function),
                                              std::move(funcSpec.argTypes)},
                        parentFrame);
                }
            }

            auto returnType = pylir::get<pylir::Py::ObjectTypeInterface>(std::move(continuation));
            auto [spec, _] = info.insert(
                {currentExecutionStack.back().getMonomorphFunction().getFunctionOp(),
                 std::move(currentExecutionStack.back()).getMonomorphFunction().getArgTypes()},
                FinishedAnalysis{returnType,
                                 std::move(currentExecutionStack.back()).getMonomorphFunction().getLattices()});
            if (currentExecutionStack.back().getParentFrame() < currentExecutionStack.size())
            {
                currentExecutionStack[currentExecutionStack.back().getParentFrame()].addCallResult(returnType);
            }
            currentlyExecuting.erase(spec->first);
            currentExecutionStack.pop_back();
        }
    }

    struct Clone
    {
        mlir::Operation* clone;
        mlir::BlockAndValueMapping mapping;
    };

    llvm::DenseMap<FunctionSpecialization, Clone> functionClones;
    for (const auto& [key, value] : info.getCalculatedSpecializations())
    {
        auto funcOp = mlir::cast<mlir::FunctionOpInterface>(key.function);
        if (llvm::equal(funcOp.getArgumentTypes(), key.argTypes))
        {
            if (funcOp.getNumResults() == 1 && funcOp.getResultTypes()[0] != value.returnType)
            {
                m_typesRefined++;
                funcOp.setType(mlir::FunctionType::get(&getContext(), funcOp.getArgumentTypes(), value.returnType));
            }
            continue;
        }

        m_functionsCloned++;
        mlir::BlockAndValueMapping mapping;
        auto clone = mlir::cast<mlir::FunctionOpInterface>(funcOp->clone(mapping));
        symbolTable.insert(clone);
        llvm::SmallVector<mlir::Type> argTypes(key.argTypes.begin(), key.argTypes.end());
        clone.setType(mlir::FunctionType::get(&getContext(), argTypes, value.returnType));
        for (auto [blockArg, newType] : llvm::zip(clone.front().getArguments(), key.argTypes))
        {
            if (blockArg.getType() != newType)
            {
                blockArg.setType(newType);
                m_typesRefined++;
            }
        }
        functionClones.insert({key, {clone, std::move(mapping)}});
    }

    for (const auto& [key, value] : info.getCalculatedSpecializations())
    {
        mlir::BlockAndValueMapping* mapping = nullptr;
        {
            auto result = functionClones.find(key);
            if (result != functionClones.end())
            {
                mapping = &result->second.mapping;
            }
        }
        for (const auto& [value, lattice] : value.lattices)
        {
            auto valueToUse = value;
            if (mapping)
            {
                valueToUse = mapping->lookup(valueToUse);
            }
            if (valueToUse.getType() == lattice.type)
            {
                continue;
            }
            valueToUse.setType(lattice.type);
            m_typesRefined++;
        }
    }
}

} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createMonomorphPass()
{
    return std::make_unique<Monomorph>();
}
