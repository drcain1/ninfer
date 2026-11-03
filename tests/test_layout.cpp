#include "core/layout.h"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Fn>
void overflows(Fn&& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) { return; }
    throw std::runtime_error("expected workspace size overflow");
}

} // namespace

int main() {
    try {
        using namespace ninfer;
        const auto limit = std::numeric_limits<std::size_t>::max();
        WorkspaceLayoutBuilder bytes;
        (void)bytes.alloc_bytes(3, 1);
        overflows([&] { (void)bytes.alloc_bytes(limit, 8); });
        require(bytes.peak_bytes(1) == 3, "failed byte layout changed the peak");
        (void)bytes.alloc_bytes(1, 1);
        require(bytes.peak_bytes(1) == 4, "failed byte layout retained alignment padding");
        {
            auto scope = bytes.scope();
            (void)bytes.alloc_bytes(3, 8);
            overflows([&] { (void)bytes.alloc_bytes(limit, 16); });
            require(bytes.peak_bytes(1) == 11, "failed scoped layout changed the peak");
            (void)bytes.alloc_bytes(1, 1);
            require(bytes.peak_bytes(1) == 12, "failed scoped layout changed the cursor");
        }
        (void)bytes.alloc_bytes(9, 1);
        require(bytes.peak_bytes(1) == 13, "scope did not restore the saved cursor");

        WorkspaceLayoutBuilder tensors;
        (void)tensors.alloc_bytes(limit - 14, 1);
        overflows([&] { (void)tensors.alloc(DType::BF16, {2, 2}, 8); });
        require(tensors.peak_bytes(1) == limit - 14, "failed tensor layout changed the peak");
        (void)tensors.alloc_bytes(1, 1);
        require(tensors.peak_bytes(1) == limit - 13,
                "failed tensor layout retained alignment padding");
        std::cout << "OK workspace layout\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
