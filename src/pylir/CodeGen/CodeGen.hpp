// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/FileSystem.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Util/PyBuilder.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Parser/Syntax.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/ValueReset.hpp>

#include <tuple>
#include <unordered_map>

namespace pylir
{

struct CodeGenOptions
{
    std::vector<std::string> importPaths;
    std::function<void(Diag::DiagnosticsBuilder&&)> warningCallback;

    struct LoadRequest
    {
        llvm::sys::fs::file_t handle;
        std::string qualifier;
        std::pair<std::size_t, std::size_t> location;
        Diag::Document* document;
        std::string filePath;
    };
    std::function<void(LoadRequest&&)> moduleLoadCallback;
    std::string qualifier;
};

class CodeGen
{
    CodeGenOptions m_options;
    Py::PyBuilder m_builder;
    mlir::ModuleOp m_module;
    mlir::func::FuncOp m_currentFunc;
    mlir::Region* m_currentRegion{};
    Diag::Document* m_document;
    mlir::Value m_classNamespace{};
    std::unordered_map<std::string, std::size_t> m_implNames;
    std::unordered_map<std::string_view, mlir::FlatSymbolRefAttr> m_builtinNamespace;

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

    struct Identifier
    {
        enum Kind
        {
            Global = 0,
            StackAlloc = 1,
            Cell = 2
        };
        std::variant<mlir::Operation*, SSABuilder::DefinitionsMap, mlir::Value> kind;
    };

    struct Scope
    {
        std::unordered_map<std::string_view, Identifier> identifiers;
        SSABuilder ssaBuilder;
    };

    Scope m_globalScope;
    std::optional<Scope> m_functionScope;
    std::string m_qualifiers;

    [[nodiscard]] auto markOpenBlock(mlir::Block* block)
    {
        getCurrentScope().ssaBuilder.markOpenBlock(block);
        return llvm::make_scope_exit([this, block] { getCurrentScope().ssaBuilder.sealBlock(block); });
    }

    Scope& getCurrentScope()
    {
        return m_functionScope ? *m_functionScope : m_globalScope;
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
            if (!m_block->hasNoPredecessors() && m_block->getParent())
            {
                return;
            }
            m_block->dropAllReferences();
            if (m_block->getParent())
            {
                m_block->erase();
                return;
            }
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

        [[nodiscard]] mlir::Block* get() const
        {
            PYLIR_ASSERT(m_block);
            return m_block;
        }

        // NOLINTNEXTLINE(google-explicit-constructor)
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

    void raiseException(mlir::Value exceptionObject);

    mlir::Value buildSubclassCheck(mlir::Value type, mlir::Value base);

    void buildTupleForEach(mlir::Value tuple, mlir::Block* endBlock, mlir::ValueRange endArgs,
                           llvm::function_ref<void(mlir::Value)> iterationCallback);

    mlir::Value makeTuple(const std::vector<Py::IterArg>& args);

    mlir::Value makeList(const std::vector<Py::IterArg>& args);

    mlir::Value makeSet(const std::vector<Py::IterArg>& args);

    mlir::Value makeDict(const std::vector<Py::DictArg>& args);

    struct ModuleSpec
    {
        std::size_t dots;
        std::pair<std::size_t, std::size_t> dotsLocation;

        struct Component
        {
            std::string name;
            std::pair<std::size_t, std::size_t> location;
        };
        std::vector<Component> components;

        explicit ModuleSpec(const Syntax::ImportStmt::Module& module);

        explicit ModuleSpec(const Syntax::ImportStmt::RelativeModule& relativeModule);
    };

    struct ModuleImport
    {
        std::string moduleSymbolName;
        bool successful;
        std::pair<std::size_t, std::size_t> location;
    };

    std::vector<ModuleImport> importModules(llvm::ArrayRef<ModuleSpec> specs);

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

    struct UnpackResults
    {
        mlir::Value parameterValue;
        mlir::Value parameterSet;
    };

    std::vector<UnpackResults> unpackArgsKeywords(mlir::Value tuple, mlir::Value dict,
                                                  const std::vector<FunctionParameter>& parameters,
                                                  llvm::function_ref<mlir::Value(std::size_t)> posDefault = {},
                                                  llvm::function_ref<mlir::Value(llvm::StringRef)> kwDefault = {});

    mlir::func::FuncOp buildFunctionCC(llvm::Twine name, mlir::func::FuncOp implementation,
                                       const std::vector<FunctionParameter>& parameters);

    Py::GlobalValueOp createGlobalConstant(Py::ObjectAttrInterface value);

    using SlotMapImpl = std::map<std::string_view, std::variant<mlir::FlatSymbolRefAttr, mlir::SymbolOpInterface>>;

    Py::GlobalValueOp createClass(mlir::FlatSymbolRefAttr className,
                                  llvm::MutableArrayRef<Py::GlobalValueOp> bases = {},
                                  llvm::function_ref<void(SlotMapImpl&)> implementation = {});

    Py::GlobalValueOp createFunction(llvm::StringRef functionName, const std::vector<FunctionParameter>& parameters,
                                     llvm::function_ref<void(mlir::Value, mlir::ValueRange)> implementation = {},
                                     mlir::func::FuncOp* implOut = nullptr, Py::TupleAttr posArgs = {},
                                     Py::DictAttr kwArgs = {});

    Py::GlobalValueOp createFunction(llvm::StringRef functionName, const std::vector<FunctionParameter>& parameters,
                                     llvm::function_ref<void(mlir::ValueRange)> implementation,
                                     mlir::func::FuncOp* implOut = nullptr, Py::TupleAttr posArgs = {},
                                     Py::DictAttr kwArgs = {});

    Py::GlobalValueOp createExternal(llvm::StringRef objectName);

    void binCheckOtherOp(mlir::Value other, const Builtins::Builtin& builtin);

    std::vector<UnpackResults> createOverload(const std::vector<FunctionParameter>& parameters, mlir::Value tuple,
                                              mlir::Value dict, Py::TupleAttr posArgs = {}, Py::DictAttr kwArgs = {});

    template <class AST>
    mlir::Location getLoc(const AST& astObject)
    {
        auto [line, col] = m_document->getLineCol(Diag::pointLoc(astObject));
        return mlir::OpaqueLoc::get(
            &astObject, mlir::FileLineColLoc::get(m_builder.getStringAttr(m_document->getFilename()), line, col));
    }

    template <class AST, class LineLoc>
    mlir::Location getLoc(const AST& astObject, const LineLoc& lineLoc)
    {
        auto [line, col] = m_document->getLineCol(Diag::pointLoc(lineLoc));
        return mlir::OpaqueLoc::get(
            &astObject, mlir::FileLineColLoc::get(m_builder.getStringAttr(m_document->getFilename()), line, col));
    }

    std::string formImplName(std::string_view symbol);

    template <class T, std::enable_if_t<is_abstract_variant_concrete<T>{}>* = nullptr>
    void assignTarget(const T& variant, mlir::Value value)
    {
        variant.match([=](const auto& sub) { assignTarget(sub, value); });
    }

    void visit(llvm::function_ref<void(mlir::Value)> insertOperation, const Syntax::Expression& iteration,
               const Syntax::CompFor& compFor);

    void visit(llvm::function_ref<void(mlir::Value)> insertOperation, const Syntax::Expression& iteration,
               const Syntax::CompIf& compIf);

    void visit(llvm::function_ref<void(mlir::Value)> insertOperation, const Syntax::Comprehension& comprehension);

    bool needsTerminator()
    {
        return m_builder.getBlock()
               && (m_builder.getBlock()->empty()
                   || !m_builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>());
    }

    void createBuiltinsImpl();

    void createCompilerBuiltinsImpl();

    void implementBlock(mlir::Block* block)
    {
        m_currentRegion->push_back(block);
        m_builder.setInsertionPointToStart(block);
    }

    void implementBlock(const BlockPtr& blockPtr)
    {
        implementBlock(blockPtr.get());
    }

    void visitForConstruct(const Syntax::Target& targets, mlir::Value iterable, llvm::function_ref<void()> execSuite,
                           const std::optional<Syntax::IfStmt::Else>& elseSection = {});

    template <class T, std::enable_if_t<is_abstract_variant_concrete<T>{}>* = nullptr>
    decltype(auto) visit(const T& variant)
    {
        auto lambda = [&] { return variant.match([=](const auto& sub) -> decltype(auto) { return visit(sub); }); };
        using Ret = decltype(lambda());
        if (!m_builder.getInsertionBlock())
        {
            if constexpr (std::is_void_v<Ret>)
            {
                return;
            }
            else
            {
                return Ret{};
            }
        }
        auto currLoc = m_builder.getCurrentLoc();
        auto exit = llvm::make_scope_exit([=] { m_builder.setCurrentLoc(currLoc); });
        if constexpr (Diag::hasLocationProvider_v<T>)
        {
            m_builder.setCurrentLoc(getLoc(variant));
        }
        return lambda();
    }

    template <class T, std::enable_if_t<!std::is_convertible_v<T, mlir::Location>>* = nullptr>
    [[nodiscard]] auto changeLoc(const T& astNode)
    {
        auto currLoc = m_builder.getCurrentLoc();
        auto exit = llvm::make_scope_exit([=] { m_builder.setCurrentLoc(currLoc); });
        m_builder.setCurrentLoc(getLoc(astNode));
        return exit;
    }

    template <class T, class F>
    [[nodiscard]] auto changeLoc(const T& astNode, const F& lineLocProv)
    {
        auto currLoc = m_builder.getCurrentLoc();
        auto exit = llvm::make_scope_exit([=] { m_builder.setCurrentLoc(currLoc); });
        m_builder.setCurrentLoc(getLoc(astNode, lineLocProv));
        return exit;
    }

    [[nodiscard]] auto changeLoc(mlir::Location loc)
    {
        auto currLoc = m_builder.getCurrentLoc();
        auto exit = llvm::make_scope_exit([=] { m_builder.setCurrentLoc(currLoc); });
        m_builder.setCurrentLoc(loc);
        return exit;
    }

    [[nodiscard]] auto implementFunction(mlir::func::FuncOp funcOp)
    {
        auto tuple =
            std::make_tuple(mlir::OpBuilder::InsertionGuard(m_builder),
                            pylir::valueResetMany(m_currentFunc, m_currentRegion, m_currentLoop, m_currentExceptBlock,
                                                  std::move(m_functionScope), m_qualifiers));
        m_currentLoop = {nullptr, nullptr};
        m_currentExceptBlock = nullptr;
        m_currentFunc = funcOp;
        m_currentRegion = &m_currentFunc.getBody();
        m_module.push_back(m_currentFunc);
        m_builder.setInsertionPointToStart(m_currentFunc.addEntryBlock());
        m_functionScope.emplace(Scope{{},
                                      SSABuilder(
                                          [this](mlir::BlockArgument arg) -> mlir::Value
                                          {
                                              auto locExit = changeLoc(arg.getLoc());
                                              mlir::OpBuilder::InsertionGuard guard{m_builder};
                                              m_builder.setInsertionPointToStart(arg.getOwner());
                                              return m_builder.createConstant(m_builder.getUnboundAttr());
                                          })});
        return tuple;
    }

    template <class T, class S, class... Args>
    [[nodiscard]] Diag::DiagnosticsBuilder createDiagnosticsBuilder(const T& location, const S& message,
                                                                    Args&&... args) const
    {
        return Diag::DiagnosticsBuilder(*m_document, location, message, std::forward<Args>(args)...);
    }

public:
    CodeGen(mlir::MLIRContext* context, Diag::Document& document, CodeGenOptions&& options);

    mlir::ModuleOp visit(const Syntax::FileInput& fileInput);

    void visit(const Syntax::IfStmt& ifStmt);

    void visit(const Syntax::WhileStmt& whileStmt);

    void visit(const Syntax::ForStmt& forStmt);

    void visit(const Syntax::TryStmt& tryStmt);

    void visit(const Syntax::WithStmt& withStmt);

    void visit(const Syntax::FuncDef& funcDef);

    void visit(const Syntax::ClassDef& classDef);

    void visit(const Syntax::Suite& suite);

    void visit(const Syntax::AssignmentStmt& assignmentStmt);

    void visit(const Syntax::RaiseStmt& raiseStmt);

    void visit(const Syntax::ReturnStmt& returnStmt);

    void visit(const Syntax::SingleTokenStmt& singleTokenStmt);

    void visit(const Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt);

    void visit(const Syntax::ExpressionStmt& expressionStmt);

    void visit(const Syntax::AssertStmt& assertStmt);

    void visit(const Syntax::DelStmt& delStmt);

    void visit(const Syntax::ImportStmt& importStmt);

    void visit(const Syntax::FutureStmt& futureStmt);

    void assignTarget(const Syntax::Atom& atom, mlir::Value value);

    void assignTarget(const Syntax::Subscription& subscription, mlir::Value value);

    void assignTarget(const Syntax::AttributeRef& attributeRef, mlir::Value value);

    void assignTarget(const Syntax::TupleConstruct& tupleConstruct, mlir::Value value);

    void assignTarget(const Syntax::ListDisplay& listDisplay, mlir::Value value);

    template <class T,
              std::enable_if_t<std::is_base_of_v<Syntax::Target, T> && !std::is_same_v<Syntax::Target, T>>* = nullptr>
    void assignTarget(const T&, mlir::Value)
    {
        PYLIR_UNREACHABLE;
    }

    mlir::Value visit(const Syntax::Yield& yield);

    mlir::Value visit(const Syntax::Conditional& expression);

    mlir::Value visit(const Syntax::Comparison& comparison);

    mlir::Value visit(const Syntax::Atom& atom);

    mlir::Value visit(const Syntax::Subscription& primary);

    mlir::Value visit(const Syntax::Assignment& assignment);

    mlir::Value visit(const Syntax::TupleConstruct& tupleConstruct);

    mlir::Value visit(const Syntax::BinOp& binOp);

    mlir::Value visit(const Syntax::UnaryOp& unaryOp);

    mlir::Value visit(const Syntax::AttributeRef& attributeRef);

    mlir::Value visit(const Syntax::Slice& slice);

    mlir::Value visit(const Syntax::Call& call);

    mlir::Value visit(const Syntax::Lambda& lambda);

    mlir::Value visit(const Syntax::Generator& generator);

    mlir::Value visit(const Syntax::ListDisplay& listDisplay);

    mlir::Value visit(const Syntax::SetDisplay& setDisplay);

    mlir::Value visit(const Syntax::DictDisplay& dictDisplay);

    std::vector<Py::IterArg> visit(llvm::ArrayRef<Syntax::StarredItem> starredItems);

    std::pair<mlir::Value, mlir::Value> visit(llvm::ArrayRef<Syntax::Argument> argumentList);
};

inline mlir::OwningOpRef<mlir::ModuleOp> codegen(mlir::MLIRContext* context, const Syntax::FileInput& input,
                                                 Diag::Document& document, CodeGenOptions options)
{
    CodeGen codegen(context, document, std::move(options));
    return codegen.visit(input);
}

} // namespace pylir
