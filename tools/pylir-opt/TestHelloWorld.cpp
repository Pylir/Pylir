#include "Passes.hpp"

#include <iostream>

namespace
{
class TestHelloWorld : public TestHelloWorldBase<TestHelloWorld>
{
protected:
    void runOnOperation() override
    {
        // Using cout here as it has an atomicity guarantee for single invocations of operator<<
        std::cout << "Hello World!\n";
    }
};
} // namespace

std::unique_ptr<mlir::Pass> createTestHelloWorld()
{
    return std::make_unique<TestHelloWorld>();
}
