
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/MlirOptMain.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Mem::PylirDialect, pylir::Py::PylirPyDialect>();

    pylir::registerConversionPasses();

    return mlir::failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
