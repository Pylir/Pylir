//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/FileSystem.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Parser/Syntax.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/ValueReset.hpp>

#include <tuple>
#include <unordered_map>

#include "PyBuilder.hpp"

namespace pylir
{

struct CodeGenOptions
{
    std::vector<std::string> importPaths;

    struct LoadRequest
    {
        llvm::sys::fs::file_t handle;
        std::string qualifier;
        std::pair<std::size_t, std::size_t> location;
        Diag::DiagnosticsDocManager* diagnosticsDocManager;
        std::string filePath;
    };
    std::function<void(LoadRequest&&)> moduleLoadCallback;
    std::string qualifier;
};

class CodeGen
{
    CodeGenOptions m_options;
    PyBuilder m_builder;
    mlir::ModuleOp m_module;
    mlir::func::FuncOp m_currentFunc;
    mlir::Region* m_currentRegion{};
    Diag::DiagnosticsDocManager* m_docManager;
    mlir::Value m_classNamespace{};
    std::unordered_map<std::string, std::size_t> m_implNames;
    std::unordered_map<std::string_view, Py::RefAttr> m_builtinNamespace;
    bool m_constantClass = false;

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
            Local = 1,
            Cell = 2
        };
        std::variant<Py::GlobalOp, SSABuilder::DefinitionsMap, mlir::Value> kind;
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

    bool inGlobalScope() const
    {
        return !m_functionScope;
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

    mlir::Value readIdentifier(std::string_view name);

    void writeIdentifier(std::string_view text, mlir::Value value);

    void raiseException(mlir::Value exceptionObject);

    mlir::Value buildSubclassCheck(mlir::Value type, mlir::Value base);

    void buildTupleForEach(mlir::Value tuple, mlir::Block* endBlock, mlir::ValueRange endArgs,
                           llvm::function_ref<void(mlir::Value)> iterationCallback);

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

        explicit ModuleSpec(std::vector<Component> components)
            : dots{}, dotsLocation{}, components(std::move(components))
        {
        }
    };

    struct ModuleImport
    {
        std::string moduleSymbolName;
        bool successful;
        std::pair<std::size_t, std::size_t> location;
    };

    std::vector<ModuleImport> importModules(llvm::ArrayRef<ModuleSpec> specs);

    struct Intrinsic
    {
        std::string name;
        std::vector<IdentifierToken> identifiers;
    };

    std::optional<Intrinsic> checkForIntrinsic(const Syntax::Expression& expression);

    mlir::Value callIntrinsic(Intrinsic&& intrinsic, llvm::ArrayRef<Syntax::Argument> arguments,
                              const Syntax::Call& call);

    mlir::Value intrinsicConstant(Intrinsic&& intrinsic);

    std::optional<bool> checkDecoratorIntrinsics(llvm::ArrayRef<Syntax::Decorator> decorators,
                                                 bool additionalConstCondition);

    template <class T>
    T dereference(mlir::Attribute attr)
    {
        if (auto val = attr.dyn_cast_or_null<T>())
        {
            return val;
        }
        auto ref = attr.dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
        if (!ref)
        {
            return nullptr;
        }
        auto globalValue = m_module.lookupSymbol<Py::GlobalValueOp>(ref);
        return globalValue.getInitializerAttr().dyn_cast_or_null<T>();
    }

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

    template <class AST>
    mlir::Location getLoc(const AST& astObject)
    {
        auto [line, col] = m_docManager->getDocument().getLineCol(Diag::pointLoc(astObject));
        return mlir::OpaqueLoc::get(
            &astObject,
            mlir::FileLineColLoc::get(m_builder.getStringAttr(m_docManager->getDocument().getFilename()), line, col));
    }

    template <class AST, class LineLoc>
    mlir::Location getLoc(const AST& astObject, const LineLoc& lineLoc)
    {
        auto [line, col] = m_docManager->getDocument().getLineCol(Diag::pointLoc(lineLoc));
        return mlir::OpaqueLoc::get(
            &astObject,
            mlir::FileLineColLoc::get(m_builder.getStringAttr(m_docManager->getDocument().getFilename()), line, col));
    }

    std::string formImplName(std::string_view symbol);

    template <class T, std::enable_if_t<is_abstract_variant_concrete<T>{}>* = nullptr>
    void assignTarget(const T& variant, mlir::Value value)
    {
        variant.match([=](const auto& sub) { assignTarget(sub, value); });
    }

    template <class T, std::enable_if_t<is_abstract_variant_concrete<T>{}>* = nullptr>
    void delTarget(const T& variant)
    {
        variant.match([=](const auto& sub) { delTarget(sub); });
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

    mlir::Value visitFunction(llvm::ArrayRef<Syntax::Decorator> decorators,
                              llvm::ArrayRef<Syntax::Parameter> parameterList, llvm::StringRef funcName,
                              const Syntax::Scope& scope, llvm::function_ref<void()> emitFunctionBody);

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

    [[nodiscard]] mlir::Location synthesizedLoc()
    {
        return mlir::FileLineColLoc::get(m_builder.getStringAttr(m_docManager->getDocument().getFilename()), 0, 0);
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
                                          [this](mlir::Block* block, mlir::Type, mlir::Location loc) -> mlir::Value
                                          {
                                              auto locExit = changeLoc(loc);
                                              mlir::OpBuilder::InsertionGuard guard{m_builder};
                                              m_builder.setInsertionPointToStart(block);
                                              return m_builder.createConstant(m_builder.getUnboundAttr());
                                          })});
        return tuple;
    }

public:
    CodeGen(mlir::MLIRContext* context, Diag::DiagnosticsDocManager& docManager, CodeGenOptions&& options);

    template <class T, class S, class... Args, std::enable_if_t<Diag::hasLocationProvider_v<T>>* = nullptr>
    auto createError(const T& location, const S& message, Args&&... args)
    {
        return Diag::DiagnosticsBuilder(*m_docManager, Diag::Severity::Error, location, message,
                                        std::forward<Args>(args)...);
    }

    template <class T, class S, class... Args, std::enable_if_t<Diag::hasLocationProvider_v<T>>* = nullptr>
    auto createWarning(const T& location, const S& message, Args&&... args)
    {
        return Diag::DiagnosticsBuilder(*m_docManager, Diag::Severity::Warning, location, message,
                                        std::forward<Args>(args)...);
    }

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

    void assignTarget(const Syntax::Slice& slice, mlir::Value value);

    void assignTarget(const Syntax::AttributeRef& attributeRef, mlir::Value value);

    void assignTarget(llvm::ArrayRef<Syntax::StarredItem> starredItems, mlir::Value value);

    void assignTarget(const Syntax::TupleConstruct& tupleConstruct, mlir::Value value);

    void assignTarget(const Syntax::ListDisplay& listDisplay, mlir::Value value);

    template <class T, std::enable_if_t<std::is_base_of_v<Syntax::Target, T> && !std::is_same_v<Syntax::Target, T>
                                        && !Syntax::validTargetType<T>()>* = nullptr>
    void assignTarget(const T&, mlir::Value)
    {
        PYLIR_UNREACHABLE;
    }

    void delTarget(const Syntax::Atom& atom);

    void delTarget(const Syntax::Subscription& subscription);

    void delTarget(const Syntax::Slice& slice);

    void delTarget(const Syntax::AttributeRef& attributeRef);

    void delTarget(const Syntax::TupleConstruct& tupleConstruct);

    void delTarget(const Syntax::ListDisplay& listDisplay);

    template <class T, std::enable_if_t<std::is_base_of_v<Syntax::Target, T> && !std::is_same_v<Syntax::Target, T>
                                        && !Syntax::validTargetType<T>()>* = nullptr>
    void delTarget(const T&)
    {
        PYLIR_UNREACHABLE;
    }

    mlir::Value visit(const Syntax::Yield& yield);

    mlir::Value visit(const Syntax::Conditional& conditional);

    mlir::Value visit(const Syntax::Comparison& comparison);

    mlir::Value visit(const Syntax::Atom& atom);

    mlir::Value visit(const Syntax::Subscription& subscription);

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

mlir::Value buildException(mlir::Location loc, PyBuilder& builder, std::string_view kind, std::vector<Py::IterArg> args,
                           mlir::Block* PYLIR_NULLABLE exceptionHandler);

inline mlir::OwningOpRef<mlir::ModuleOp> codegen(mlir::MLIRContext* context, const Syntax::FileInput& input,
                                                 Diag::DiagnosticsDocManager& docManager, CodeGenOptions options)
{
    CodeGen codegen(context, docManager, std::move(options));
    return codegen.visit(input);
}

} // namespace pylir
