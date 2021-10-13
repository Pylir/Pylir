
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/MlirOptMain.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transform/Passes.hpp>

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Mem::PylirDialect, pylir::Py::PylirPyDialect>();
    mlir::registerAllDialects(registry);

    pylir::registerConversionPasses();
    pylir::Py::registerTransformPasses();

    return mlir::failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
