//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Block.h>

#include <pylir/Support/Macros.hpp>

#include <functional>

#include "ValueTracker.hpp"

namespace pylir {
/// This is an implementation of:
/// Braun, M., Buchwald, S., Hack, S., Leißa, R., Mallon, C., Zwinkau, A.
/// (2013). Simple and Efficient Construction of Static Single Assignment Form.
/// In: Jhala, R., De Bosschere, K. (eds) Compiler Construction. CC 2013.
/// Lecture Notes in Computer Science, vol 7791. Springer, Berlin, Heidelberg.
/// https://doi.org/10.1007/978-3-642-37051-9_6
///
/// It is a simple algorithm for constructing SSA representation and is capable
/// of doing so online. Its use is in code generation to be able to create SSA
/// representation of bindings during AST traversal as well as in optimization
/// phases, where it can be used to create SSA representation at any point in
/// time.
///
/// One requirement deferred to the user is marking so called 'open' blocks.
/// 'Open' blocks are blocks in the CFG that do not yet know all their
/// predecessors. These blocks have to marked using 'markOpenBlock'. As soon as
/// all predecessors of an 'open' block have been created, a call to 'sealBlock'
/// has to be made. During AST traversal, this is very easily done. The only
/// blocks created without yet having all known predecessors at a specific point
/// in time are loop headers. One therefore has to mark loop headers as open
/// upon creation and seal them as soon as all loop back-edges have been created
/// (aka. after the loop body has been generated). In the case of optimization
/// passes one simply has to memorize all so far processed blocks during
/// traversal and mark a block as open if not all its predecessors have been
/// processed. See 'pylir::updateSSAinRegion' in 'SSAUpdater.hpp' for a utility
/// function that does this automatically.
///
/// Defs and Uses:
/// The SSABuilder class requires any user to keep track of a
/// 'SSABuilder::DefinitionsMap' per variable that should be transformed into
/// SSA. This map is a simple mapping from a block to latest definition of a
/// variable in a block. Any definitions to the variable are therefore not
/// performed by the SSABuilder, but instead done by updating the map with the
/// new value of the variable inside of the block. To create a use of the
/// variable, 'readVariable' should be called, which will automatically create
/// all required block arguments and return the current value of the variable in
/// the block.
///
/// SSABuilder currently only supports pure block based CFG and no subregions.
/// All terminators with successors are required to implement
/// 'BranchOpInterface'.
class SSABuilder {
  using InternalDefinitionsMap = llvm::DenseMap<mlir::Block*, ValueTracker>;

public:
  /// The definition map users use to create new definitions of a variable
  /// within a block. It's a simple wrapper around a DenseMap, mapping blocks to
  /// ValueTrackers, keeping its address stable for the sake of capturing by the
  /// SSABuilder.
  class DefinitionsMap {
    std::unique_ptr<llvm::DenseMap<mlir::Block*, ValueTracker>> m_map;

  public:
    using value_type = typename std::decay_t<decltype(*m_map)>::value_type;

    DefinitionsMap()
        : m_map(
              std::make_unique<llvm::DenseMap<mlir::Block*, ValueTracker>>()) {}

    explicit DefinitionsMap(llvm::DenseMap<mlir::Block*, ValueTracker>&& map)
        : m_map(std::make_unique<llvm::DenseMap<mlir::Block*, ValueTracker>>(
              std::move(map))) {}

    explicit DefinitionsMap(
        const llvm::DenseMap<mlir::Block*, ValueTracker>& map)
        : m_map(std::make_unique<llvm::DenseMap<mlir::Block*, ValueTracker>>(
              map)) {}

    DefinitionsMap(std::initializer_list<value_type> initList)
        : m_map(std::make_unique<llvm::DenseMap<mlir::Block*, ValueTracker>>(
              initList)) {}

    ValueTracker& operator[](mlir::Block* block) {
      return (*m_map)[block];
    }

    auto insert(value_type pair) {
      return m_map->insert(std::move(pair));
    }

    auto find(mlir::Block* block) {
      return m_map->find(block);
    }

    auto find(mlir::Block* block) const {
      return m_map->find(block);
    }

    auto end() {
      return m_map->end();
    }

    [[nodiscard]] auto end() const {
      return m_map->end();
    }

    InternalDefinitionsMap* get() {
      return m_map.get();
    }

    InternalDefinitionsMap& operator*() {
      return *m_map;
    }
  };

private:
  llvm::DenseMap<mlir::Block*, std::vector<InternalDefinitionsMap*>>
      m_openBlocks;
  std::function<mlir::Value(mlir::Block*, mlir::Type, mlir::Location)>
      m_undefinedCallback;
  std::function<mlir::Value(mlir::Value, mlir::Value)>
      m_blockArgMergeOptCallback;
  llvm::DenseMap<mlir::Block*, mlir::BlockArgument> m_marked;

  // Checks if all the operands can be optimized to a single value (either
  // because they're all the same value or equal to 'maybeArgument' etc.) and if
  // so returns it. Otherwise, returns nullptr. 'maybeArgument' is optional and
  // if set is the block argument that has 'operands' as inputs. 'block', 'type'
  // and 'loc' are used to call 'm_undefinedCallback'.
  mlir::Value optimizeBlockArgsOperands(llvm::ArrayRef<mlir::Value> operands,
                                        mlir::BlockArgument maybeArgument,
                                        mlir::Block* block, mlir::Type type,
                                        mlir::Location loc);

  mlir::Value tryRemoveTrivialBlockArgument(mlir::BlockArgument argument);

  mlir::Value replaceBlockArgument(mlir::BlockArgument argument,
                                   mlir::Value replacement);

  mlir::Value addBlockArguments(InternalDefinitionsMap& map,
                                mlir::BlockArgument argument);

  mlir::Value readVariableRecursive(mlir::Location loc, mlir::Type type,
                                    InternalDefinitionsMap& map,
                                    mlir::Block* block);

  void removeBlockArgumentOperands(mlir::BlockArgument argument);

  mlir::Value readVariable(mlir::Location loc, mlir::Type type,
                           InternalDefinitionsMap& map, mlir::Block* block);

public:
  /// Creates a new SSA builder.
  /// The 'undefinedCallback' is called in the case that a use for a variable is
  /// generated that does not yet have any definitions. It is called with a
  /// block dominating the use which the return value will be replacing. If no
  /// 'undefinedCallback' is passed, it'll assert instead.
  ///
  /// The 'blockArgMergeOptCallback' is an optional callback that can be used to
  /// optimize away a block argument by calculating a value through a fold over
  /// all its operands. It is called for each operand to the block argument
  /// where 'curr' is the current fold result and 'argOp' is the next operand
  /// leading into the block argument. On first call 'curr' is initialized to
  /// the very first operand and 'argOp' will be the second.
  explicit SSABuilder(
      std::function<mlir::Value(mlir::Block*, mlir::Type, mlir::Location)>
          undefinedCallback =
              [](auto&&...) -> mlir::Value { PYLIR_UNREACHABLE; },
      std::function<mlir::Value(mlir::Value curr, mlir::Value argOp)>
          blockArgMergeOptCallback = {})
      : m_undefinedCallback(std::move(undefinedCallback)),
        m_blockArgMergeOptCallback(std::move(blockArgMergeOptCallback)) {}

  ~SSABuilder() {
    PYLIR_ASSERT(m_openBlocks.empty());
  }

  SSABuilder(const SSABuilder&) = delete;
  SSABuilder& operator=(const SSABuilder&) = delete;
  SSABuilder(SSABuilder&&) noexcept = default;
  SSABuilder& operator=(SSABuilder&&) noexcept = default;

  /// Returns true if 'block' is an open block.
  bool isOpenBlock(mlir::Block* block) const {
    return m_openBlocks.count(block);
  }

  /// Marks a block as being an open block.
  void markOpenBlock(mlir::Block* block);

  /// Seals a block. Does nothing if the block is not an open block.
  void sealBlock(mlir::Block* block);

  /// Creates a use of a variable. 'loc' and 'type' are used to create any
  /// required block arguments in predecessors blocks. 'type' also has to match
  /// the type of the variable being created. 'map' is the "block to latest
  /// definition" map handled by the user. 'block' is the block where the use is
  /// being created.
  ///
  /// NOTE: 'map' may be captured in the case of encountering a seal block and
  /// must therefore outlive the SSABuilder.
  ///       It does not have to have a stable address however.
  mlir::Value readVariable(mlir::Location loc, mlir::Type type,
                           DefinitionsMap& map, mlir::Block* block) {
    return readVariable(loc, type, *map, block);
  }
};
} // namespace pylir
