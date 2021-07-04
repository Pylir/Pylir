
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Support/MlirOptMain.h>

#include <pylir/Dialect/PylirDialect.hpp>

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Dialect::PylirDialect>();
    registry.insert<mlir::scf::SCFDialect>();

    return mlir::failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
