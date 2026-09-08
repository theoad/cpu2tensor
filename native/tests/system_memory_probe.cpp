// SPDX-License-Identifier: AGPL-3.0-only
// Deliberately slow diagnostic plugin, never part of the capture runtime.
#include <qemu-plugin.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

extern "C" {
QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
}

namespace {
constexpr unsigned kContextCount = 4;
constexpr uint64_t kGuestStart = 0x7c00;
constexpr uint64_t kGuestEnd = 0x7e00;
const char* kNames[kContextCount] = {"cr0", "cr3", "cr4", "efer"};
qemu_plugin_register* registers[kContextCount] = {};
GByteArray* scratch = nullptr;
uint64_t block_context[kContextCount] = {};
unsigned events = 0;
unsigned mismatches = 0;
using StateHook = bool (*)(uint64_t*, size_t);
StateHook state_hook = nullptr;
unsigned seen_widths = 0;
bool raw_compatibility_seen = false;

void ReadContext(uint64_t* context) {
    for (unsigned i = 0; i < kContextCount; ++i) {
        g_byte_array_set_size(scratch, 0);
        if (registers[i] == nullptr ||
            !qemu_plugin_read_register(registers[i], scratch) ||
            scratch->len == 0 || scratch->len > sizeof(uint64_t)) {
            std::fprintf(stderr, "probe: cannot read %s\n", kNames[i]);
            std::abort();
        }
        context[i] = 0;
        for (unsigned byte = 0; byte < scratch->len; ++byte) {
            context[i] |= uint64_t(scratch->data[byte]) << (8 * byte);
        }
    }
}

void Block(unsigned, void*) {
    ReadContext(block_context);
    if (state_hook == nullptr) {
        return;
    }
    uint64_t full[28];
    uint64_t context[8];
    if (!state_hook(full, 28) || !state_hook(context, 8) ||
        std::memcmp(full, context, sizeof(context)) != 0) {
        std::fprintf(stderr, "probe: hook context/full mismatch\n");
        std::abort();
    }
    const unsigned bit = full[7] == 16 ? 1 : full[7] == 32 ? 2 : full[7] == 64 ? 4 : 0;
    const bool raw_compatibility = full[7] == 32 &&
        full[8] == UINT64_C(0x1122334455667788) &&
        full[16] == UINT64_C(0x8877665544332211);
    if ((seen_widths & bit) == 0 || raw_compatibility) {
        std::printf("hook width=%" PRIu64 " csbase=%" PRIx64
                    " eip=%" PRIx64 " rax=%" PRIx64 " r8=%" PRIx64 "\n",
                    full[7], full[6], full[24], full[8], full[16]);
    }
    seen_widths |= bit;
    raw_compatibility_seen |= raw_compatibility;
}

void NoRegisterAccess(unsigned, void*) {
    if (state_hook != nullptr) {
        uint64_t guard[8] = {};
        if (state_hook(guard, 8)) {
            std::fprintf(stderr, "probe: hook accepted a NO_REGS callback\n");
            std::abort();
        }
    }
}

void Memory(unsigned, qemu_plugin_meminfo_t info, uint64_t address, void*) {
    if (address != 0x8ffe && address != 0xaffe && address != 0xb030) {
        return;
    }
    uint64_t context[kContextCount];
    ReadContext(context);
    const bool equal = std::memcmp(context, block_context, sizeof(context)) == 0;
    mismatches += !equal;
    if (state_hook != nullptr) {
        uint64_t state[8];
        if (!state_hook(state, 8) || state[0] != context[0] ||
            state[2] != context[1] || state[3] != context[2] ||
            state[5] != context[3]) {
            std::fprintf(stderr, "probe: hook/public control mismatch\n");
            std::abort();
        }
    }
    auto* first = qemu_plugin_get_hwaddr(info, address);
    const uint64_t physical = qemu_plugin_hwaddr_phys_addr(first);
    const bool is_io = first && qemu_plugin_hwaddr_is_io(first);
    const unsigned width = 1u << qemu_plugin_mem_size_shift(info);
    const auto value = qemu_plugin_mem_get_value(info);
    // This extra lookup tests implementation behavior. The public header only
    // documents passing the address supplied to the memory callback.
    auto* second = address == 0x8ffe ?
        qemu_plugin_get_hwaddr(info, address + 2) : nullptr;
    const uint64_t second_physical = qemu_plugin_hwaddr_phys_addr(second);
    const uint64_t number = width == 2 ? value.data.u16 : value.data.u32;
    std::printf("event=%u store=%u va=%" PRIx64 " width=%u value=%" PRIx64
                " first=%" PRIx64 " second=%" PRIx64 " io=%u"
                " cr0=%" PRIx64 " cr3=%" PRIx64 " cr4=%" PRIx64
                " efer=%" PRIx64 " same_context=%u\n",
                events++, qemu_plugin_mem_is_store(info), address, width, number,
                physical, second_physical, is_io, context[0], context[1],
                context[2], context[3], equal);
    std::fflush(stdout);
}

void Translate(qemu_plugin_id_t, qemu_plugin_tb* block) {
    const auto pc = qemu_plugin_tb_vaddr(block);
    if (pc < kGuestStart || pc >= kGuestEnd) {
        return;
    }
    qemu_plugin_register_vcpu_tb_exec_cb(block, Block, QEMU_PLUGIN_CB_R_REGS, nullptr);
    qemu_plugin_register_vcpu_tb_exec_cb(block, NoRegisterAccess,
                                        QEMU_PLUGIN_CB_NO_REGS, nullptr);
    for (size_t i = 0; i < qemu_plugin_tb_n_insns(block); ++i) {
        qemu_plugin_register_vcpu_mem_cb(qemu_plugin_tb_get_insn(block, i),
            Memory, QEMU_PLUGIN_CB_R_REGS, QEMU_PLUGIN_MEM_RW, nullptr);
    }
}

void Initialize(qemu_plugin_id_t, unsigned) {
    scratch = g_byte_array_sized_new(64);
    GArray* descriptors = qemu_plugin_get_registers();
    for (unsigned i = 0; i < descriptors->len; ++i) {
        const auto& item = g_array_index(descriptors, qemu_plugin_reg_descriptor, i);
        for (unsigned j = 0; j < kContextCount; ++j) {
            if (std::strcmp(item.name, kNames[j]) == 0) {
                registers[j] = item.handle;
            }
        }
    }
    g_array_free(descriptors, true);
    if (state_hook != nullptr) {
        uint64_t guard = UINT64_C(0xfedcba9876543210);
        if (state_hook(nullptr, 8) || state_hook(&guard, 1) ||
            guard != UINT64_C(0xfedcba9876543210)) {
            std::fprintf(stderr, "probe: hook invalid arguments accepted\n");
            std::abort();
        }
    }
}

void Finish(qemu_plugin_id_t, void*) {
    std::printf("finished events=%u context_mismatches=%u\n", events, mismatches);
    if (state_hook != nullptr) {
        std::printf("hook seen_widths=%u raw_compatibility=%u\n",
                    seen_widths, raw_compatibility_seen);
    }
}
}  // namespace

extern "C" QEMU_PLUGIN_EXPORT int qemu_plugin_install(
    qemu_plugin_id_t id, const qemu_info_t* info, int, char**) {
    if (!info->system_emulation || info->system.smp_vcpus != 1) {
        std::fprintf(stderr, "probe needs one system vCPU\n");
        return -1;
    }
    state_hook = reinterpret_cast<StateHook>(
        dlsym(RTLD_DEFAULT, "qemu_plugin_cpu2tensor_x86_state_v1"));
    qemu_plugin_register_vcpu_init_cb(id, Initialize);
    qemu_plugin_register_vcpu_tb_trans_cb(id, Translate);
    qemu_plugin_register_atexit_cb(id, Finish, nullptr);
    return 0;
}
