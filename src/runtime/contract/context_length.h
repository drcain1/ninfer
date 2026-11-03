#pragma once

#include <cstddef>
#include <string>

namespace ninfer::runtime {

// Bounded tokenization and partial media planning provide a lower bound on prompt length.
[[nodiscard]] inline std::string context_length_error_message(std::size_t tokens,
                                                              std::size_t max_context,
                                                              bool lower_bound = false) {
    return "maximum context length is " + std::to_string(max_context) +
           " tokens, but the prepared prompt has " + (lower_bound ? "at least " : "") +
           std::to_string(tokens) + " tokens";
}

} // namespace ninfer::runtime
