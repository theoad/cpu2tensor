// SPDX-License-Identifier: AGPL-3.0-only
#include "kernel_protocol.hpp"
#include <cassert>
#include <fcntl.h>

using namespace cpu2tensor::kernel_protocol;

static void test_json_classification() {
    struct Case { const char* payload; const char* problem; };
    const Case cases[] = {
        {"{\"event\":\"ready\",\"step\":0}", nullptr},
        {"{\"event\":\"result\"}\r\n", nullptr},
        {"{", "incomplete-json"},
        {"{broken}", "malformed-json"},
        {"{\"event\":\"ready\"}garbage", "malformed-json"},
        {"[]", "non-object-json"},
        {"\"ready\"", "non-object-json"},
        {"null ", "non-object-json"},
        {"{}", "missing-event-field"},
        {"{\"event\":null}", "non-string-event-field"},
        {"{\"event\":false}", "non-string-event-field"},
        {"{\"event\":42}", "non-string-event-field"},
        {"{\"event\":{}}", "non-string-event-field"},
        {"{\"event\":\"ready\\u0000extra\"}", "embedded-null-event-field"},
    };
    for (const auto& item : cases) {
        const Json parsed(item.payload, std::strlen(item.payload));
        if (item.problem == nullptr) {
            assert(parsed.event_problem() == nullptr);
            assert(string_value(parsed.get("event")) != nullptr);
        } else {
            assert(parsed.event_problem() != nullptr);
            assert(std::strcmp(parsed.event_problem(), item.problem) == 0);
        }
        assert(parsed.parse_end() <= std::strlen(item.payload));
    }
    // A real NUL must not hide an appended record or count as a string terminator.
    const char with_null[] = "{\"event\":\"ready\"}\0hidden";
    const Json nul(with_null, sizeof(with_null) - 1);
    assert(nul.event_problem() != nullptr);
}

static void test_escaped_diagnostic() {
    char payload[event_capacity + 100];
    std::memset(payload, '\033', sizeof(payload));
    payload[0] = '"';
    payload[1] = '\\';
    payload[2] = '\n';
    payload[3] = '\r';
    payload[4] = '\0';
    FILE* output = std::tmpfile();
    assert(output != nullptr);
    const Json parsed(payload, sizeof(payload));
    log_guest_failure(output, parsed.event_problem(), payload, sizeof(payload), &parsed);
    assert(std::fflush(output) == 0);
    assert(std::fseek(output, 0, SEEK_SET) == 0);
    char message[4 * event_capacity + 512]{};
    const auto size = std::fread(message, 1, sizeof(message) - 1, output);
    assert(std::feof(output));
    assert(size < sizeof(message) - 1);
    assert(std::strstr(message, "bytes=1124 shown=1024 truncated=true") != nullptr);
    assert(std::strstr(message, "parse_end=") != nullptr);
    assert(std::strstr(message, "root_type=") != nullptr);
    assert(std::strstr(message, "\\\"\\\\\\x0a\\x0d\\x00\\x1b") != nullptr);
    assert(message[size - 1] == '\n');
    for (size_t i = 0; i + 1 < size; ++i)
        assert(message[i] >= 32 && message[i] <= 126);
    std::fclose(output);

    output = std::tmpfile();
    assert(output != nullptr);
    std::memset(payload, 'x', event_capacity);
    log_guest_failure(output, "test", payload, event_capacity);
    std::rewind(output);
    std::memset(message, 0, sizeof(message));
    assert(std::fread(message, 1, sizeof(message) - 1, output) > 0);
    assert(std::strstr(message, "bytes=1024 shown=1024 truncated=false") != nullptr);
    const char* preview = std::strstr(message, "payload=\"");
    assert(preview != nullptr);
    preview += std::strlen("payload=\"");
    for (size_t i = 0; i < event_capacity; ++i) assert(preview[i] == 'x');
    assert(std::strcmp(preview + event_capacity, "\"\n") == 0);
    std::fclose(output);
}

static void test_fragmented_and_coalesced_lines() {
    constexpr char records[] = "C2T {\"event\":\"result\"}\r\nC2T {\"event\":\"ready\"}\n";
    const size_t first = static_cast<size_t>(std::strchr(records, '\n') - records) + 1;
    // Exercise every socket read split, including inside CRLF and between records.
    for (size_t split = 1; split < sizeof(records) - 1; ++split) {
        int descriptors[2];
        assert(pipe(descriptors) == 0);
        Lines lines;
        assert(write(descriptors[1], records, split) == static_cast<ssize_t>(split));
        assert(lines.read_from(descriptors[0]).ok());
        size_t consumed = 0;
        if (lines.line_size() != 0) {
            assert(lines.line_size() == first);
            lines.consume(first);
            consumed = first;
        }
        assert(write(descriptors[1], records + split, sizeof(records) - 1 - split) ==
               static_cast<ssize_t>(sizeof(records) - 1 - split));
        assert(lines.read_from(descriptors[0]).ok());
        while (const size_t size = lines.line_size()) {
            assert(std::memcmp(lines.bytes, records + consumed, size) == 0);
            const Json event(lines.bytes + 4, size - 4);
            assert(event.event_problem() == nullptr);
            lines.consume(size);
            consumed += size;
        }
        assert(consumed == sizeof(records) - 1);
        assert(lines.used == 0);
        close(descriptors[1]);
        assert(lines.read_from(descriptors[0]).ok() && lines.eof);
        close(descriptors[0]);
    }
    Lines full;
    full.used = line_capacity - 1;
    assert(!full.read_from(-1).ok());
    int descriptors[2];
    assert(pipe(descriptors) == 0);
    assert(write(descriptors[1], "C2T {", 5) == 5);
    close(descriptors[1]);
    Lines partial;
    assert(partial.read_from(descriptors[0]).ok());
    assert(partial.read_from(descriptors[0]).ok() && partial.eof);
    assert(partial.used == 5 && partial.line_size() == 0);
    close(descriptors[0]);
}

static void test_guest_payload_framing() {
    constexpr char json[] =
        "{\"event\":\"result\",\"step\":2,\"action\":\"getpid\",\"value\":1}";
    constexpr char printk[] =
        "[    6.795768] input: ImExPS/2 Generic Explorer Mouse as "
        "/devices/platform/i8042/serio1/input/input3";
    char observed[sizeof(json) + sizeof(printk) - 1];
    std::memcpy(observed, json, sizeof(json) - 1);
    std::memcpy(observed + sizeof(json) - 1, printk, sizeof(printk));
    const auto recovered = split_guest_payload(observed, sizeof(observed) - 1);
    assert(recovered.json_size == sizeof(json) - 1);
    assert(recovered.suffix == GuestSuffix::kernel_printk);
    assert(recovered.suffix_offset == sizeof(json) - 1);
    assert(recovered.suffix_size == sizeof(printk) - 1);
    const Json event(observed, recovered.json_size);
    assert(event.event_problem() == nullptr);

    constexpr char nested[] =
        "{\"event\":\"result\",\"nested\":{\"text\":\"} C2T \\\" [\"}}"
        "[0.000001] nested braces stayed inside the JSON string";
    const auto nested_split = split_guest_payload(nested, sizeof(nested) - 1);
    const Json nested_event(nested, nested_split.json_size);
    assert(nested_event.event_problem() == nullptr);
    assert(nested_split.suffix == GuestSuffix::kernel_printk);

    constexpr char arbitrary[] = "{\"event\":\"result\"}garbage";
    assert(split_guest_payload(arbitrary, sizeof(arbitrary) - 1).suffix ==
           GuestSuffix::invalid);
    constexpr char second[] =
        "{\"event\":\"result\"}C2T {\"event\":\"ready\"}";
    assert(split_guest_payload(second, sizeof(second) - 1).suffix == GuestSuffix::invalid);
    constexpr char hidden_second[] =
        "{\"event\":\"result\"}[0.000001] message contains C2T {\"event\":\"ready\"}";
    assert(split_guest_payload(hidden_second, sizeof(hidden_second) - 1).suffix ==
           GuestSuffix::invalid);
    constexpr char empty_printk[] = "{\"event\":\"result\"}[0.000001] ";
    assert(split_guest_payload(empty_printk, sizeof(empty_printk) - 1).suffix ==
           GuestSuffix::invalid);
    constexpr char malformed[] = "{broken}[0.000001] diagnostic";
    const auto malformed_split = split_guest_payload(malformed, sizeof(malformed) - 1);
    const Json malformed_event(malformed, malformed_split.json_size);
    assert(malformed_event.event_problem() != nullptr);
    constexpr char ordinary[] = "{\"event\":\"result\"}\t ";
    const auto ordinary_split = split_guest_payload(ordinary, sizeof(ordinary) - 1);
    assert(ordinary_split.json_size == sizeof(ordinary) - 1);
    assert(ordinary_split.suffix == GuestSuffix::none);
}

int main() {
    test_json_classification();
    test_escaped_diagnostic();
    test_fragmented_and_coalesced_lines();
    test_guest_payload_framing();
}
