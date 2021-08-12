
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/MlirOptMain.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/Dialect/PylirDialect.hpp>

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Dialect::PylirDialect>();

    pylir::Dialect::registerConversionPasses();

    return mlir::failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
