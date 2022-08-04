//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/GenNameParser.h>

#include <llvm/Support/InitLLVM.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>

// Below is taken from
// https://github.com/llvm/llvm-project/blob/e4fb75a354740bf45dab0ebd43f37ab2fdeae3bf/mlir/tools/mlir-tblgen/mlir-tblgen.cpp#L40-L39
// with only adjustments being stylistic aka non-functioning changes (code formatting)

//===- mlir-tblgen.cpp - Top-Level TableGen implementation for MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static llvm::ManagedStatic<std::vector<mlir::GenInfo>> generatorRegistry;

mlir::GenRegistration::GenRegistration(StringRef arg, StringRef description, const GenFunction& function)
{
    generatorRegistry->emplace_back(arg, description, function);
}

mlir::GenNameParser::GenNameParser(llvm::cl::Option& opt) : llvm::cl::parser<const GenInfo*>(opt)
{
    for (const auto& kv : *generatorRegistry)
    {
        addLiteralOption(kv.getGenArgument(), &kv, kv.getGenDescription());
    }
}

void mlir::GenNameParser::printOptionInfo(const llvm::cl::Option& o, size_t globalWidth) const
{
    auto* tp = const_cast<GenNameParser*>(this);
    llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                         [](const GenNameParser::OptionInfo* vT1, const GenNameParser::OptionInfo* vT2)
                         { return vT1->Name.compare(vT2->Name); });
    using llvm::cl::parser;
    parser<const GenInfo*>::printOptionInfo(o, globalWidth);
}

// Generator to invoke.
const mlir::GenInfo* generator;

static bool mlirTableGenMain(llvm::raw_ostream& os, llvm::RecordKeeper& records)
{
    if (!generator)
    {
        os << records;
        return false;
    }
    return generator->invoke(records, os);
}

int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);
    llvm::cl::opt<const mlir::GenInfo*, false, mlir::GenNameParser> generator("", llvm::cl::desc("Generator to run"));
    llvm::cl::ParseCommandLineOptions(argc, argv);
    ::generator = generator.getValue();

    return TableGenMain(argv[0], &mlirTableGenMain);
}
