#include "Passes.hpp"

namespace
{
class TestHelloWorld : public TestHelloWorldBase<TestHelloWorld>
{
protected:
    void runOnOperation() override
    {
        llvm::outs() << "Hello World!\n";
    }
};
} // namespace

std::unique_ptr<mlir::Pass> createTestHelloWorld()
{
    return std::make_unique<TestHelloWorld>();
}
