#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Parser/Syntax.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/ValueReset.hpp>

#include <tuple>
#include <unordered_map>

namespace pylir
{

namespace Builtins
{

struct Builtin
{
    std::string_view name;
    bool isPublic;
};

#define BUILTIN(x, s, isPublic) constexpr Builtin x = {s, isPublic};
#include "Builtins.def"

constexpr std::array allBuiltins = {
#define BUILTIN(x, ...) x,
#include "Builtins.def"
};
} // namespace Builtins

class CodeGen
{
    mlir::OpBuilder m_builder;
    mlir::ModuleOp m_module;
    mlir::FuncOp m_currentFunc;
    Diag::Document* m_document;
    mlir::Value m_classNamespace{};
    std::vector<std::string> m_qualifierStack;
    std::unordered_map<std::string, std::size_t> m_implNames;
    std::unordered_map<std::string_view, std::string_view> m_builtinNamespace;

    struct Loop
    {
        mlir::Block* breakBlock;
        mlir::Block* continueBlock;

        bool operator==(const Loop& rhs) const
        {
            return std::tie(breakBlock, continueBlock) == std::tie(rhs.breakBlock, rhs.continueBlock);
        }

        bool operator!=(const Loop& rhs) const
        {
            return !(rhs == *this);
        }
    } m_currentLoop{nullptr, nullptr};

    mlir::Block* m_currentExceptBlock = nullptr;
    struct FinallyBlocks
    {
        const Syntax::TryStmt::Finally* PYLIR_NON_NULL finallySuite;
        Loop parentLoop;
        mlir::Block* parentExceptBlock;
    };
    std::vector<FinallyBlocks> m_finallyBlocks;

    void executeFinallyBlocks(bool fullUnwind = false);

    enum Kind
    {
        Global,
        StackAlloc,
        Cell
    };

    struct Identifier
    {
        Kind kind;
        mlir::PointerUnion<mlir::Value, mlir::Operation*> op;
    };

    using ScopeContainer = std::vector<std::unordered_map<std::string_view, Identifier>>;
    ScopeContainer m_scope{1};

    std::unordered_map<std::string_view, Identifier>& getCurrentScope()
    {
        return m_scope.back();
    }

    class BlockPtr
    {
        mlir::Block* m_block;

        void maybeDestroy()
        {
            if (!m_block)
            {
                return;
            }
            if (!m_block->hasNoPredecessors())
            {
                PYLIR_ASSERT(m_block->getParent() && "Block has uses but does not have a parent");
                return;
            }
            PYLIR_ASSERT(!m_block->getParent() && "Block should not have a parent region if it is unreachable");
            delete m_block;
        }

    public:
        BlockPtr() : m_block(new mlir::Block) {}

        explicit BlockPtr(mlir::Block* block) : m_block(block) {}

        ~BlockPtr()
        {
            maybeDestroy();
        }

        BlockPtr(const BlockPtr&) = delete;
        BlockPtr& operator=(const BlockPtr&) = delete;
        BlockPtr(BlockPtr&& rhs) noexcept : m_block(std::exchange(rhs.m_block, nullptr)) {}
        BlockPtr& operator=(BlockPtr&& rhs) noexcept
        {
            maybeDestroy();
            m_block = std::exchange(rhs.m_block, nullptr);
            return *this;
        }

        mlir::Block* get() const
        {
            PYLIR_ASSERT(m_block);
            return m_block;
        }

        operator mlir::Block*() const
        {
            return get();
        }

        mlir::Block* operator->() const
        {
            return get();
        }

        friend bool operator==(const BlockPtr& lhs, mlir::Block* rhs)
        {
            return lhs.get() == rhs;
        }

        friend bool operator==(mlir::Block* rhs, const BlockPtr& lhs)
        {
            return lhs.get() == rhs;
        }
    };

    mlir::Value toI1(mlir::Value value);

    mlir::Value toBool(mlir::Value value);

    mlir::Value readIdentifier(const IdentifierToken& token);

    void writeIdentifier(const IdentifierToken& token, mlir::Value value);

    mlir::Value buildException(mlir::Location loc, std::string_view type, std::vector<Py::IterArg> args);

    void raiseException(mlir::Value exceptionObject);

    mlir::Value buildCall(mlir::Location loc, mlir::Value callable, mlir::Value tuple, mlir::Value dict);

    std::pair<mlir::Value, mlir::Value> buildMROLookup(mlir::Location loc, mlir::Value type, llvm::Twine attribute);

    mlir::Value buildSubclassCheck(mlir::Location loc, mlir::Value type, mlir::Value base);

    void buildTupleForEach(mlir::Location loc, mlir::Value tuple, mlir::Block* endBlock, mlir::ValueRange endArgs,
                           llvm::function_ref<void(mlir::Value)> iterationCallback);

    mlir::Value buildSpecialMethodCall(mlir::Location loc, llvm::Twine methodName, mlir::Value type, mlir::Value tuple,
                                       mlir::Value dict);

    mlir::Value makeTuple(mlir::Location loc, const std::vector<Py::IterArg>& args);

    mlir::Value makeList(mlir::Location loc, const std::vector<Py::IterArg>& args);

    mlir::Value makeSet(mlir::Location loc, const std::vector<Py::IterArg>& args);

    mlir::Value makeDict(mlir::Location loc, const std::vector<Py::DictArg>& args);

    struct FunctionParameter
    {
        std::string name;
        enum Kind
        {
            Normal,
            PosOnly,
            KeywordOnly,
            PosRest,
            KeywordRest,
        } kind;
        bool hasDefaultParam;
    };

    mlir::FuncOp buildFunctionCC(mlir::Location loc, llvm::Twine name, mlir::FuncOp implementation,
                                 const std::vector<FunctionParameter>& parameters);

    template <class AST, class FallBackLocation>
    mlir::Location getLoc(const AST& astObject, const FallBackLocation& fallBackLocation)
    {
        auto [line, col] = m_document->getLineCol(Diag::range(fallBackLocation).first);
        return mlir::OpaqueLoc::get(
            &astObject, mlir::FileLineColLoc::get(m_builder.getIdentifier(m_document->getFilename()), line, col));
    }

    std::string formQualifiedName(std::string_view symbol);

    std::string formImplName(std::string_view symbol);

    void assignTarget(const Syntax::TargetList& targetList, mlir::Value value);

    void assignTarget(const Syntax::Target& target, mlir::Value value);

    template <mlir::Value (CodeGen::*op)(mlir::Location, const std::vector<Py::IterArg>&)>
    mlir::Value visit(const Syntax::StarredList& starredList);

    bool needsTerminator()
    {
        return m_builder.getBlock()
               && (m_builder.getBlock()->empty()
                   || !m_builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>());
    }

    void createBuiltinsImpl();

    [[nodiscard]] auto implementFunction(mlir::FuncOp funcOp)
    {
        // We are putting InsertionGuard into a unique_ptr to make it moveable and not call its destructor early
        // when compiling with MSVC
        auto tuple = std::make_tuple(pylir::ValueReset(m_currentFunc),
                                     std::make_unique<mlir::OpBuilder::InsertionGuard>(m_builder),
                                     pylir::ValueReset(m_currentLoop), pylir::ValueReset(m_currentExceptBlock));
        m_currentLoop = {nullptr, nullptr};
        m_currentExceptBlock = nullptr;
        m_currentFunc = funcOp;
        m_module.push_back(m_currentFunc);
        m_builder.setInsertionPointToStart(m_currentFunc.addEntryBlock());
        return tuple;
    }

    void implementBlock(mlir::Block* block)
    {
        m_currentFunc.push_back(block);
        m_builder.setInsertionPointToStart(block);
    }

    void implementBlock(const BlockPtr& blockPtr)
    {
        implementBlock(blockPtr.get());
    }

public:
    CodeGen(mlir::MLIRContext* context, Diag::Document& document);

    mlir::ModuleOp visit(const Syntax::FileInput& fileInput);

    void visit(const Syntax::Statement& statement);

    void visit(const Syntax::StmtList& stmtList);

    void visit(const Syntax::CompoundStmt& compoundStmt);

    void visit(const Syntax::IfStmt& ifStmt);

    void visit(const Syntax::WhileStmt& whileStmt);

    void visit(const Syntax::ForStmt& forStmt);

    void visit(const Syntax::TryStmt& tryStmt);

    void visit(const Syntax::WithStmt& withStmt);

    void visit(const Syntax::FuncDef& funcDef);

    void visit(const Syntax::ClassDef& classDef);

    void visit(const Syntax::AsyncForStmt& asyncForStmt);

    void visit(const Syntax::AsyncWithStmt& asyncWithStmt);

    void visit(const Syntax::Suite& suite);

    void visit(const Syntax::SimpleStmt& continueStmt);

    void visit(const Syntax::AssignmentStmt& assignmentStmt);

    mlir::Value visit(const Syntax::StarredExpression& starredExpression);

    mlir::Value visit(const Syntax::ExpressionList& expressionList);

    mlir::Value visit(const Syntax::YieldExpression& yieldExpression);

    mlir::Value visit(const Syntax::Expression& expression);

    mlir::Value visit(const Syntax::ConditionalExpression& expression);

    mlir::Value visit(const Syntax::OrTest& expression);

    mlir::Value visit(const Syntax::AndTest& expression);

    mlir::Value visit(const Syntax::NotTest& expression);

    mlir::Value visit(const Syntax::Comparison& comparison);

    mlir::Value visit(const Syntax::OrExpr& orExpr);

    mlir::Value visit(const Syntax::XorExpr& xorExpr);

    mlir::Value visit(const Syntax::AndExpr& andExpr);

    mlir::Value visit(const Syntax::ShiftExpr& shiftExpr);

    mlir::Value visit(const Syntax::AExpr& aExpr);

    mlir::Value visit(const Syntax::MExpr& mExpr);

    mlir::Value visit(const Syntax::UExpr& uExpr);

    mlir::Value visit(const Syntax::Power& power);

    mlir::Value visit(const Syntax::AwaitExpr& awaitExpr);

    mlir::Value visit(const Syntax::Primary& primary);

    mlir::Value visit(const Syntax::Atom& atom);

    mlir::Value visit(const Syntax::Subscription& primary);

    mlir::Value visit(const Syntax::Enclosure& enclosure);

    mlir::Value visit(const Syntax::AssignmentExpression& assignmentExpression);

    std::pair<mlir::Value, mlir::Value> visit(const Syntax::ArgumentList& argumentList);
};

inline mlir::OwningOpRef<mlir::ModuleOp> codegen(mlir::MLIRContext* context, const Syntax::FileInput& input,
                                                 Diag::Document& document)
{
    CodeGen codegen{context, document};
    return codegen.visit(input);
}

} // namespace pylir
