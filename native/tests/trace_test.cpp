// SPDX-License-Identifier: AGPL-3.0-only
#include <cassert>
#include <cstring>
#include <cpu2tensor/trace.hpp>
#include <cpu2tensor/register_selection.hpp>
#include <cpu2tensor/transition_window.hpp>

using namespace cpu2tensor;

int main() {
    assert(valid_register_selection("rax:rip"));
    assert(!valid_register_selection("rax::rip"));
    assert(!valid_register_selection("rax,memory=off"));
    assert(!valid_register_selection(":rax"));
    assert(!valid_register_selection("rax:"));
    assert(listed_register("rax:rip", "rip"));
    assert(!listed_register("rax:rip", "ra"));
    assert(unavailable_x86_register("ftag", false));
    assert(unavailable_x86_register("eflags", true));
    assert(!unavailable_x86_register("eflags", false));
    uint8_t bytes[header_bytes];
    Header batch{Kind::blocks, 2, 2, 0, 0};
    encode_header(bytes, batch);
    assert(std::memcmp(bytes, "C2T1\3\0\2\0", 8) == 0);
    assert(decode_header(bytes, sizeof(bytes)).value().source == 2);
    bytes[4] = 4;
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    assert(!decode_header(bytes, 4).ok());
    store_u64(bytes, UINT64_MAX);
    assert(load_u64(bytes) == UINT64_MAX);
    store_u64(bytes, uint64_t{1} << 63);
    assert(load_u64(bytes) == uint64_t{1} << 63);
    assert(payload_size({Kind::hello}) == 0);

    Header legacy_hello{Kind::hello, 0, 0, 0, 1};
    legacy_hello.version = legacy_wire_version;
    encode_header(bytes, legacy_hello);
    assert(decode_header(bytes, sizeof(bytes)).ok());
    Stream legacy_stream;
    assert(legacy_stream.accept(legacy_hello).ok());
    Header legacy_end{Kind::source_end};
    legacy_end.version = legacy_wire_version;
    assert(legacy_stream.accept(legacy_end).ok());
    Header legacy_complete{Kind::complete};
    legacy_complete.version = legacy_wire_version;
    assert(legacy_stream.accept(legacy_complete).ok());

    const auto terminal = terminal_report_detail(
        TerminalReason::max_run_deadline,
        terminal_hello | terminal_data | terminal_start_configured |
        terminal_start_observed | terminal_stop_configured);
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0, terminal});
    assert(decode_header(bytes, sizeof(bytes)).ok());
    Header legacy_terminal{Kind::terminal_report, 0, 0, 0, terminal};
    legacy_terminal.version = legacy_wire_version;
    encode_header(bytes, legacy_terminal);
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0, terminal | (uint64_t{1} << 24)});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0, terminal + 1});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0, terminal + (uint64_t{1} << 8)});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::terminal_report, 0, 0, 0,
                          terminal_report_detail(TerminalReason::max_run_deadline,
                                                 terminal_start_observed)});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    Stream terminal_only;
    assert(terminal_only.accept({Kind::terminal_report, 0, 0, 0, terminal_report_detail(
        TerminalReason::max_run_deadline, 0)}).ok());
    assert(terminal_only.finished());
    assert(!terminal_only.accept({Kind::hello, 0, 0, 0, 1}).ok());
    Stream mixed_versions;
    assert(mixed_versions.accept(legacy_hello).ok());
    assert(!mixed_versions.accept({Kind::source_end}).ok());

    // Exercise every accepted compound frame shape as well as combinations
    // that would give a client ambiguous trace semantics.
    encode_header(bytes, {Kind::mixed, 0, 1, 0, 16});
    assert(decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::executable_layout, 0, 1, 0, layout_bytes});
    assert(decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::transition_window, 0, 1, 1, transition_window_bytes});
    assert(decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::context_filter, 0, 1, 0, context_filter_bytes});
    assert(decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::memory, 0, 1, 0, 24});
    assert(decode_header(bytes, sizeof(bytes)).ok());

    encode_header(bytes, {Kind::input_request, 1, 0, 0, 1});
    assert(!decode_header(bytes, sizeof(bytes)).ok());

    uint8_t filter[context_filter_bytes]{};
    store_u32(filter, 1);
    store_u32(filter + 4, 2);
    store_u64(filter + 16, 0x81001000);
    store_u64(filter + 24, 0x12000);
    store_u64(filter + 32, 0x12000);
    store_u64(filter + 40, 2);
    store_u64(filter + 48, 3);
    store_u64(filter + 56, 2);
    store_u64(filter + 64, 2);
    store_u64(filter + 72, 1);
    uint8_t filter_context[context_bytes]{};
    store_u64(filter_context + 56, 15);
    uint8_t filter_memory[48 * 2]{};
    store_u32(filter_memory + 16, 4);
    store_u64(filter_memory + 40, 0);
    store_u32(filter_memory + 48 + 16, 4);
    store_u64(filter_memory + 48 + 40, 0);
    Stream filtered;
    assert(filtered.accept({Kind::hello, 0, 0, 0,
                            2 | feature_system | feature_memory | feature_system_memory |
                            feature_address_context | feature_context_filter}).ok());
    assert(filtered.accept({Kind::address_context, 0, 1, 0, context_bytes}, filter_context).ok());
    assert(filtered.accept({Kind::memory, 0, 2, 1, sizeof(filter_memory)}, filter_memory).ok());
    assert(filtered.accept({Kind::source_end, 0, 0, 3, 0}).ok());
    assert(!filtered.accept({Kind::complete}).ok());
    assert(filtered.accept({Kind::context_filter, 0, 1, 0, context_filter_bytes}, filter).ok());
    assert(filtered.accept({Kind::complete}).ok());

    // A mixed memory subrun contributes to the same accepted-row total. The
    // producer summary must close that total instead of being trusted alone.
    uint8_t mixed_memory[8 + 48]{};
    store_u16(mixed_memory, static_cast<uint16_t>(Kind::memory));
    store_u16(mixed_memory + 2, 1);
    store_u32(mixed_memory + 4, 48);
    store_u32(mixed_memory + 8 + 16, 4);
    store_u64(mixed_memory + 8 + 40, 0);
    uint8_t one_kept[context_filter_bytes]{};
    store_u32(one_kept, 1);
    store_u32(one_kept + 4, 2);
    store_u64(one_kept + 16, 0x81001000);
    store_u64(one_kept + 24, 0x12000);
    store_u64(one_kept + 32, 0x12000);
    store_u64(one_kept + 40, 1);
    store_u64(one_kept + 56, 1);
    Stream filtered_mixed;
    assert(filtered_mixed.accept({Kind::hello, 0, 0, 0,
                                  2 | feature_system | feature_memory | feature_system_memory |
                                  feature_address_context | feature_mixed |
                                  feature_context_filter}).ok());
    assert(filtered_mixed.accept({Kind::address_context, 0, 1, 0, context_bytes},
                                  filter_context).ok());
    assert(filtered_mixed.accept({Kind::mixed, 0, 1, 1, sizeof(mixed_memory)},
                                  mixed_memory).ok());
    assert(filtered_mixed.accept({Kind::source_end, 0, 0, 2, 0}).ok());
    assert(filtered_mixed.accept({Kind::context_filter, 0, 1, 0, context_filter_bytes},
                                  one_kept).ok());

    uint8_t wrong_kept[context_filter_bytes];
    std::memcpy(wrong_kept, one_kept, sizeof(wrong_kept));
    store_u64(wrong_kept + 40, 0);
    store_u64(wrong_kept + 48, 1);
    store_u64(wrong_kept + 56, 0);
    store_u64(wrong_kept + 64, 1);
    Stream filtered_wrong_count;
    assert(filtered_wrong_count.accept({Kind::hello, 0, 0, 0,
                                        2 | feature_system | feature_memory |
                                        feature_system_memory | feature_address_context |
                                        feature_mixed | feature_context_filter}).ok());
    assert(filtered_wrong_count.accept({Kind::address_context, 0, 1, 0, context_bytes},
                                        filter_context).ok());
    assert(filtered_wrong_count.accept({Kind::mixed, 0, 1, 1, sizeof(mixed_memory)},
                                        mixed_memory).ok());
    assert(filtered_wrong_count.accept({Kind::source_end, 0, 0, 2, 0}).ok());
    assert(!filtered_wrong_count.accept({Kind::context_filter, 0, 1, 0,
                                         context_filter_bytes}, wrong_kept).ok());

    uint8_t absent_gate[context_filter_bytes];
    std::memcpy(absent_gate, one_kept, sizeof(absent_gate));
    store_u32(absent_gate + 8, 1);
    Stream filtered_absent_gate;
    assert(filtered_absent_gate.accept({Kind::hello, 0, 0, 0,
                                        2 | feature_system | feature_memory |
                                        feature_system_memory | feature_address_context |
                                        feature_context_filter}).ok());
    assert(filtered_absent_gate.accept({Kind::address_context, 0, 1, 0, context_bytes},
                                        filter_context).ok());
    assert(filtered_absent_gate.accept({Kind::memory, 0, 1, 1, 48}, filter_memory).ok());
    assert(filtered_absent_gate.accept({Kind::source_end, 0, 0, 2, 0}).ok());
    assert(!filtered_absent_gate.accept({Kind::context_filter, 0, 1, 0,
                                         context_filter_bytes}, absent_gate).ok());
    encode_header(bytes, {Kind::complete, 1, 0, 0, 0});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::hello, 0, 0, 0,
                          2 | feature_system | feature_memory | feature_system_memory});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::hello, 0, 0, 0, 2 | feature_address_context});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::hello, 0, 0, 0,
                          2 | feature_system | feature_registers});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {Kind::hello, 0, 0, 0, 2 | feature_transition_windows});
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    encode_header(bytes, {static_cast<Kind>(UINT16_MAX), 0, 0, 0, 0});
    assert(!decode_header(bytes, sizeof(bytes)).ok());

    Stream disabled_adapter;
    assert(disabled_adapter.accept({Kind::hello, 0, 0, 0, 2}).ok());
    assert(!disabled_adapter.accept({Kind::kernel_request, 0, 0, 0, 1}).ok());
    Stream missing_guest_event;
    assert(missing_guest_event.accept({Kind::hello, 0, 0, 0,
                                       2 | feature_system | feature_kernel}).ok());
    assert(!missing_guest_event.accept({Kind::guest_event, 0, 0, 0, 1}).ok());
    Stream multi_source_stdio;
    assert(multi_source_stdio.accept({Kind::hello, 0, 0, 0, 1 | feature_stdio}).ok());
    assert(!multi_source_stdio.accept({Kind::blocks, 1, 1, 0, 0}).ok());
    Stream disabled_stdio;
    assert(disabled_stdio.accept({Kind::hello, 0, 0, 0, 1}).ok());
    assert(!disabled_stdio.accept({Kind::input_request, 0, 0, 0, 1}).ok());

    Stream stream;
    assert(!stream.accept(batch).ok());
    assert(stream.accept({Kind::hello, 0, 0, 0, 1}).ok());
    assert(!stream.accept({Kind::hello, 0, 0, 0, 1}).ok());
    assert(stream.accept(batch).ok());
    assert(!stream.accept(batch).ok());
    // Source 1 can finish while source 2 is still active.
    assert(stream.accept({Kind::source_end, 1, 0, 0, 0}).ok());
    assert(!stream.accept({Kind::complete}).ok());
    assert(!stream.accept({Kind::blocks, 2, 1, 3, 0}).ok());
    assert(stream.accept({Kind::blocks, 2, 1, 2, 0}).ok());
    assert(stream.accept({Kind::source_end, 2, 0, 3, 0}).ok());
    assert(!stream.accept({Kind::source_end, 2, 0, 3, 0}).ok());
    assert(stream.accept({Kind::complete}).ok());
    assert(!stream.accept({Kind::complete}).ok());
    assert(stream.finished());

    uint8_t transition[transition_count_bytes];
    store_u64(transition, 10);
    store_u64(transition + 8, 20);
    store_u64(transition + 16, 3);
    uint8_t summary[transition_window_bytes]{};
    store_u32(summary, static_cast<uint32_t>(WindowStatus::ended));
    store_u32(summary + 4, 2);
    store_u32(summary + 8, 8);
    store_u64(summary + 16, 1);
    store_u64(summary + 24, 3);
    Stream windows;
    assert(windows.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                           feature_kernel | feature_transition_windows}).ok());
    assert(windows.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(windows.accept({Kind::block_transitions, 1, 1, 1,
                           transition_count_bytes}, transition).ok());
    assert(windows.accept({Kind::transition_window, 0, 1, 1,
                           transition_window_bytes}, summary).ok());
    assert(!windows.accept({Kind::transition_window, 0, 1, 1,
                            transition_window_bytes}, summary).ok());
    assert(!windows.accept({Kind::transition_window, 0, 1, 2,
                            transition_window_bytes}, summary).ok());
    assert(windows.accept({Kind::source_end, 0, 0, 0, 0}).ok());
    assert(windows.accept({Kind::source_end, 1, 0, 0, 0}).ok());
    assert(windows.accept({Kind::complete}).ok());

    Stream bad_transition_source;
    assert(bad_transition_source.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                         feature_kernel | feature_transition_windows}).ok());
    assert(!bad_transition_source.accept({Kind::block_transitions, max_sources, 1, 1,
                                          transition_count_bytes}, transition).ok());

    Stream before_first_request;
    assert(before_first_request.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                        feature_kernel | feature_transition_windows}).ok());
    assert(!before_first_request.accept({Kind::block_transitions, 0, 1, 1,
                                         transition_count_bytes}, transition).ok());

    Stream missing_window;
    assert(missing_window.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                  feature_kernel | feature_transition_windows}).ok());
    assert(missing_window.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(missing_window.accept({Kind::block_transitions, 0, 1, 1,
                                  transition_count_bytes}, transition).ok());
    assert(!missing_window.accept({Kind::complete}).ok());

    Stream missing_declared_tail;
    assert(missing_declared_tail.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                         feature_kernel | feature_transition_windows}).ok());
    assert(missing_declared_tail.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(missing_declared_tail.accept({Kind::block_transitions, 1, 1, 1,
                                         transition_count_bytes}, transition).ok());
    assert(missing_declared_tail.accept({Kind::transition_window, 0, 1, 1,
                                         transition_window_bytes}, summary).ok());
    assert(missing_declared_tail.accept({Kind::source_end, 1, 0, 0, 0}).ok());
    assert(!missing_declared_tail.accept({Kind::complete}).ok());

    store_u64(summary + 32, 1);
    Stream hidden_overflow;
    assert(hidden_overflow.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                   feature_kernel | feature_transition_windows}).ok());
    assert(hidden_overflow.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(!hidden_overflow.accept({Kind::transition_window, 0, 1, 1,
                                    transition_window_bytes}, summary).ok());

    store_u64(summary + 16, 0);
    store_u64(summary + 24, 0);
    store_u64(summary + 32, 0);
    Stream no_action;
    assert(no_action.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                             feature_kernel | feature_transition_windows}).ok());
    assert(!no_action.accept({Kind::transition_window, 0, 1, 1,
                              transition_window_bytes}, summary).ok());
    assert(no_action.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(!no_action.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());

    Stream between_actions;
    assert(between_actions.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                   feature_kernel | feature_transition_windows}).ok());
    assert(between_actions.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(between_actions.accept({Kind::transition_window, 0, 1, 1,
                                   transition_window_bytes}, summary).ok());
    assert(!between_actions.accept({Kind::block_transitions, 0, 1, 2,
                                    transition_count_bytes}, transition).ok());

    Stream exit_after_action;
    assert(exit_after_action.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                     feature_kernel | feature_transition_windows}).ok());
    assert(exit_after_action.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(exit_after_action.accept({Kind::source_end, 0, 0, 0, 0}).ok());
    assert(!exit_after_action.accept({Kind::complete}).ok());

    Stream changing_shape;
    assert(changing_shape.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                  feature_kernel | feature_transition_windows}).ok());
    assert(changing_shape.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(changing_shape.accept({Kind::transition_window, 0, 1, 1,
                                  transition_window_bytes}, summary).ok());
    assert(changing_shape.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    store_u32(summary + 4, 1);
    assert(!changing_shape.accept({Kind::transition_window, 0, 1, 2,
                                   transition_window_bytes}, summary).ok());

    uint8_t duplicate[transition_count_bytes * 2];
    std::memcpy(duplicate, transition, transition_count_bytes);
    std::memcpy(duplicate + transition_count_bytes, transition, transition_count_bytes);
    Stream duplicate_rows;
    assert(duplicate_rows.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                  feature_kernel | feature_transition_windows}).ok());
    assert(duplicate_rows.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(!duplicate_rows.accept({Kind::block_transitions, 0, 2, 1,
                                   sizeof(duplicate)}, duplicate).ok());

    // More than 256 distinct rows grows the validation table. Split them at
    // the wire payload limit to also prove uniqueness spans frame boundaries.
    constexpr uint32_t first_rows = max_payload_bytes / transition_count_bytes;
    constexpr uint32_t second_rows = 257 - first_rows;
    uint8_t first_transitions[first_rows * transition_count_bytes];
    uint8_t second_transitions[second_rows * transition_count_bytes];
    for (uint32_t row = 0; row < 257; ++row) {
        uint8_t* item = row < first_rows ?
            first_transitions + row * transition_count_bytes :
            second_transitions + (row - first_rows) * transition_count_bytes;
        store_u64(item, 1000 + row);
        store_u64(item + 8, 2000 + row);
        store_u64(item + 16, 1);
    }
    Stream grown_transition_table;
    assert(grown_transition_table.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                          feature_kernel | feature_transition_windows}).ok());
    assert(grown_transition_table.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(grown_transition_table.accept({Kind::block_transitions, 0, first_rows, 1,
                                          sizeof(first_transitions)}, first_transitions).ok());
    assert(grown_transition_table.accept({Kind::block_transitions, 0, second_rows, 1,
                                          sizeof(second_transitions)}, second_transitions).ok());
    uint8_t grown_summary[transition_window_bytes]{};
    store_u32(grown_summary, static_cast<uint32_t>(WindowStatus::ended));
    store_u32(grown_summary + 4, 1);
    store_u32(grown_summary + 8, 512);
    store_u64(grown_summary + 16, 257);
    store_u64(grown_summary + 24, 257);
    assert(grown_transition_table.accept({Kind::transition_window, 0, 1, 1,
                                          transition_window_bytes}, grown_summary).ok());

    uint8_t too_many_rows[transition_count_bytes * 3];
    for (uint32_t row = 0; row < 3; ++row) {
        store_u64(too_many_rows + row * transition_count_bytes, row + 1);
        store_u64(too_many_rows + row * transition_count_bytes + 8, row + 2);
        store_u64(too_many_rows + row * transition_count_bytes + 16, 1);
    }
    uint8_t small_summary[transition_window_bytes]{};
    store_u32(small_summary, static_cast<uint32_t>(WindowStatus::ended));
    store_u32(small_summary + 4, 1);
    store_u32(small_summary + 8, 2);
    store_u64(small_summary + 16, 3);
    store_u64(small_summary + 24, 3);
    Stream capacity_overrun;
    assert(capacity_overrun.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                    feature_kernel | feature_transition_windows}).ok());
    assert(capacity_overrun.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(capacity_overrun.accept({Kind::block_transitions, 0, 3, 1,
                                    sizeof(too_many_rows)}, too_many_rows).ok());
    assert(!capacity_overrun.accept({Kind::transition_window, 0, 1, 1,
                                     transition_window_bytes}, small_summary).ok());

    Stream ended_transition_source;
    assert(ended_transition_source.accept({Kind::hello, 0, 0, 0, 2 | feature_system |
                                           feature_kernel | feature_transition_windows}).ok());
    assert(ended_transition_source.accept({Kind::kernel_request, 0, 0, 0, 127}).ok());
    assert(ended_transition_source.accept({Kind::source_end, 0, 0, 0, 0}).ok());
    assert(!ended_transition_source.accept({Kind::block_transitions, 0, 1, 1,
                                            transition_count_bytes}, transition).ok());

    encode_header(bytes, {Kind::block_transitions, 0, max_addresses, 1,
                          max_addresses * transition_count_bytes});
    assert(!decode_header(bytes, sizeof(bytes)).ok());

    const uint64_t reduction_features = 2 | feature_system | feature_address_context |
                                        feature_observation_reduction;
    Header legacy_reduction{Kind::hello, 0, 0, 0, reduction_features};
    legacy_reduction.version = legacy_wire_version;
    encode_header(bytes, legacy_reduction);
    assert(!decode_header(bytes, sizeof(bytes)).ok());
    uint8_t reduced_context[reduced_context_bytes]{};
    store_u64(reduced_context, 0);
    store_u64(reduced_context + 8, 0x1000);
    store_u64(reduced_context + 56, 64);
    store_u64(reduced_context + 64, 63);
    uint8_t reduced_summary[observation_summary_bytes]{};
    store_u32(reduced_summary, 2);
    store_u32(reduced_summary + 4, 8);
    store_u32(reduced_summary + 8, 8);
    store_u64(reduced_summary + 16, 4);
    store_u64(reduced_summary + 24, 3);
    store_u64(reduced_summary + 32, 1);
    store_u64(reduced_summary + 48, 1);
    store_u64(reduced_summary + 56, 1);
    Stream reduced;
    assert(reduced.accept({Kind::hello, 0, 0, 0, reduction_features}).ok());
    assert(reduced.accept({Kind::reduced_context, 0, 1, 0,
                           reduced_context_bytes}, reduced_context).ok());
    assert(reduced.accept({Kind::block_transitions, 0, 1, 1,
                           transition_count_bytes}, transition).ok());
    assert(reduced.accept({Kind::observation_summary, 0, 1, 0,
                           observation_summary_bytes}, reduced_summary).ok());
    std::memset(reduced_summary + 16, 0, observation_summary_bytes - 16);
    assert(reduced.accept({Kind::observation_summary, 1, 1, 0,
                           observation_summary_bytes}, reduced_summary).ok());
    assert(reduced.accept({Kind::source_end, 0, 0, 1, 0}).ok());
    assert(reduced.accept({Kind::source_end, 1, 0, 0, 0}).ok());
    assert(reduced.accept({Kind::complete}).ok());

    Stream reduced_raw;
    assert(reduced_raw.accept({Kind::hello, 0, 0, 0, reduction_features}).ok());
    assert(!reduced_raw.accept({Kind::blocks, 0, 1, 0, 0}).ok());
    Stream reduced_missing_summary;
    assert(reduced_missing_summary.accept({Kind::hello, 0, 0, 0,
                                           reduction_features}).ok());
    assert(!reduced_missing_summary.accept({Kind::source_end, 0, 0, 0, 0}).ok());

    Stream invalid;
    assert(!invalid.accept({Kind::hello, 0, 0, 0, 99}).ok());
    assert(invalid.accept({Kind::hello, 0, 0, 0, 2}).ok());
    assert(!invalid.accept({Kind::blocks, max_sources, 1, 0, 0}).ok());
    assert(!invalid.accept({Kind::blocks, 0, max_addresses + 1, 0, 0}).ok());
    assert(invalid.accept({Kind::error, 0, 0, 0, 1}).ok());
    assert(invalid.finished());
}
