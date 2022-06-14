//
// Created by Markus Böck on 14/06/2022.
//

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
class TypeFlowMonomorph : public pylir::Py::TypeFlowMonomorphBase<TypeFlowMonomorph>
{
protected:
    void runOnOperation() override;
};

void TypeFlowMonomorph::runOnOperation() {}
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createTypeFlowMonomorphPass()
{
    return std::make_unique<TypeFlowMonomorph>();
}
