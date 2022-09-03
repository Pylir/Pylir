// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SSABuilder.hpp"

#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

void pylir::SSABuilder::markOpenBlock(mlir::Block* block)
{
    m_openBlocks.insert({block, {}});
}

void pylir::SSABuilder::sealBlock(mlir::Block* block)
{
    auto result = m_openBlocks.find(block);
    if (result == m_openBlocks.end())
    {
        return;
    }
    auto blockArgs = llvm::to_vector(block->getArguments().take_back(result->second.size()));
    for (auto [blockArgument, map] : llvm::zip(blockArgs, result->second))
    {
        addBlockArguments(*map, blockArgument);
    }
    m_openBlocks.erase(result);
}

mlir::Value pylir::SSABuilder::readVariable(mlir::Location loc, mlir::Type type, DefinitionsMap& map,
                                            mlir::Block* block)
{
    if (auto result = map.find(block); result != map.end())
    {
        return result->second;
    }
    return readVariableRecursive(loc, type, map, block);
}

void pylir::SSABuilder::removeBlockArgumentOperands(mlir::BlockArgument argument)
{
    for (auto pred = argument.getOwner()->pred_begin(); pred != argument.getOwner()->pred_end(); pred++)
    {
        auto terminator = mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
        terminator.getSuccessorOperands(pred.getSuccessorIndex()).erase(argument.getArgNumber());
    }
}

mlir::Value pylir::SSABuilder::tryRemoveTrivialBlockArgument(mlir::BlockArgument argument)
{
    mlir::Value same;
    for (auto pred = argument.getOwner()->pred_begin(); pred != argument.getOwner()->pred_end(); pred++)
    {
        mlir::Value blockOperand;
        auto terminator = mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
        auto ops = terminator.getSuccessorOperands(pred.getSuccessorIndex());
        blockOperand = ops[argument.getArgNumber()];

        if (blockOperand == same || blockOperand == argument)
        {
            continue;
        }
        if (same)
        {
            if (!m_blockArgMergeOptCallback)
            {
                return argument;
            }
            if (auto merge = m_blockArgMergeOptCallback(same, blockOperand))
            {
                same = merge;
                continue;
            }
            return argument;
        }
        same = blockOperand;
    }
    if (!same)
    {
        same = m_undefinedCallback(argument);
    }

    std::vector<mlir::BlockArgument> bas;
    for (auto& user : argument.getUses())
    {
        auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(user.getOwner());
        if (!branch)
        {
            continue;
        }
        auto ops = branch.getSuccessorBlockArgument(user.getOperandNumber());
        PYLIR_ASSERT(ops);
        if (*ops == argument)
        {
            continue;
        }
        bas.emplace_back(*ops);
    }

    removeBlockArgumentOperands(argument);
    argument.replaceAllUsesWith(same);
    argument.getOwner()->eraseArgument(argument.getArgNumber());

    for (auto ba : bas)
    {
        if (ba == same)
        {
            same = tryRemoveTrivialBlockArgument(ba);
        }
        else
        {
            tryRemoveTrivialBlockArgument(ba);
        }
    }

    return same;
}

mlir::Value pylir::SSABuilder::addBlockArguments(DefinitionsMap& map, mlir::BlockArgument argument)
{
    for (auto pred = argument.getOwner()->pred_begin(); pred != argument.getOwner()->pred_end(); pred++)
    {
        auto terminator = mlir::cast<mlir::BranchOpInterface>((*pred)->getTerminator());
        terminator.getSuccessorOperands(pred.getSuccessorIndex())
            .append(readVariable(argument.getLoc(), argument.getType(), map, *pred));
    }
    return tryRemoveTrivialBlockArgument(argument);
}

mlir::Value pylir::SSABuilder::readVariableRecursive(mlir::Location loc, mlir::Type type, DefinitionsMap& map,
                                                     mlir::Block* block)
{
    mlir::Value val;
    if (auto result = m_openBlocks.find(block); result != m_openBlocks.end())
    {
        val = block->addArgument(type, loc);
        result->second.emplace_back(&map);
    }
    else if (auto* pred = block->getUniquePredecessor())
    {
        val = readVariable(loc, type, map, pred);
    }
    else
    {
        val = block->addArgument(type, loc);
        map[block] = val;
        val = addBlockArguments(map, val.cast<mlir::BlockArgument>());
    }
    map[block] = val;
    return val;
}
