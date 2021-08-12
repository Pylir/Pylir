#include <mlir/InitAllTranslations.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Translation.h>

int main(int argc, char** argv)
{
    mlir::registerAllTranslations();

    // TODO: Register standalone translations here.

    return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
