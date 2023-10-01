//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/Interfaces/AttrVerifyInterface.hpp>
#include <pylir/Optimizer/Interfaces/SROAInterfaces.hpp>
#include <pylir/Support/BigInt.hpp>

#include <map>

#include "PylirPyAttrInterfaces.hpp"
#include "PylirPyTraits.hpp"

namespace pylir::Py::detail {
struct GlobalValueAttrStorage;
} // namespace pylir::Py::detail

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.h.inc"

template <class First, class Second>
struct mlir::AttrTypeSubElementHandler<
    std::pair<First, Second>,
    std::enable_if_t<mlir::has_sub_attr_or_type_v<First, Second>>> {
  static void walk(const std::pair<First, Second>& param,
                   AttrTypeImmediateSubElementWalker& walker) {
    AttrTypeSubElementHandler<First>::walk(param.first, walker);
    AttrTypeSubElementHandler<Second>::walk(param.second, walker);
  }

  static std::pair<First, Second>
  replace(const std::pair<First, Second>& param,
          AttrSubElementReplacements& attrRepls,
          TypeSubElementReplacements& typeRepls) {
    return {AttrTypeSubElementHandler<First>::replace(param.first, attrRepls,
                                                      typeRepls),
            AttrTypeSubElementHandler<Second>::replace(param.second, attrRepls,
                                                       typeRepls)};
  }
};
