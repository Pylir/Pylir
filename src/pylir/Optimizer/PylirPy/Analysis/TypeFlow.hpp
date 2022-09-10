//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/AnalysisManager.h>

#include "TypeFlowIR.hpp"

namespace pylir::Py
{

class TypeFlow
{
    mlir::OwningOpRef<pylir::TypeFlow::FuncOp> m_function;

public:
    explicit TypeFlow(mlir::Operation* operation);

    pylir::TypeFlow::FuncOp getFunction()
    {
        return *m_function;
    }

    void dump();
};

} // namespace pylir::Py
