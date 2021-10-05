#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Parser/Syntax.hpp>

#include <unordered_map>

namespace pylir
{
class CodeGen
{
    mlir::OpBuilder m_builder;
    mlir::ModuleOp m_module;
    mlir::FuncOp m_currentFunc;
    Diag::Document* m_document;
    mlir::Value m_classNamespace{};
    std::vector<std::string> m_qualifierStack;
    std::unordered_map<std::string, std::size_t> m_implNames;

    struct Loop
    {
        mlir::Block* breakBlock;
        mlir::Block* continueBlock;
    };
    std::vector<Loop> m_loopStack;

    enum Kind
    {
        Global,
        StackAlloc,
        Cell
    };

    struct Identifier
    {
        Kind kind;
        mlir::Operation* op;
    };

    using ScopeContainer = std::vector<std::unordered_map<std::string_view, Identifier>>;
    ScopeContainer m_scope{1};

    std::unordered_map<std::string_view, Identifier>& getCurrentScope()
    {
        return m_scope.back();
    }

    mlir::Value toI1(mlir::Value value);

    mlir::Value toBool(mlir::Value value);

    mlir::Value readIdentifier(const IdentifierToken& token);

    void writeIdentifier(const IdentifierToken& token, mlir::Value value);

    mlir::Value buildException(mlir::Location loc, Py::SingletonKind kind, std::vector<Py::IterArg> args);

    void raiseException(mlir::Value exceptionObject);

    mlir::Value buildCall(mlir::Location loc, mlir::Value callable, mlir::Value tuple, mlir::Value dict);

    std::pair<mlir::Value, mlir::Value> buildMROLookup(mlir::Location loc, mlir::Value type, llvm::Twine attribute);

    mlir::Value buildSpecialMethodCall(mlir::Location loc, llvm::Twine methodName, mlir::Value type, mlir::Value tuple,
                                       mlir::Value dict);

    struct FunctionParameter
    {
        std::string name;
        enum Kind
        {
            Normal,
            PosOnly,
            KeywordOnly,
        } kind;
        bool hasDefaultParam;
    };

    mlir::FuncOp buildFunctionCC(mlir::Location loc, llvm::StringRef name, mlir::FuncOp implementation,
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

    template <class Op>
    mlir::Value visit(const Syntax::StarredList& starredList);

    bool needsTerminator()
    {
        return m_builder.getBlock()
               && (m_builder.getBlock()->empty()
                   || !m_builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>());
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
