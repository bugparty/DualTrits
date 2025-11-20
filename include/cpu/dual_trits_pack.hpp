//
// Created by bowman on 11/8/25.
//

#ifndef PROJECT_FLOAT_PACKING_H
#define PROJECT_FLOAT_PACKING_H
#include <cstdint>
#include "common/DualTrits.hpp"
#include <limits>
#include <type_traits>


// Simple constexpr integer power
constexpr unsigned long long ipow_u(unsigned base, unsigned exp) {
    unsigned long long r = 1;
    while (exp--) r *= base;
    return r;
}

// Generic packer: pack `Count` DualTrits into unsigned integer type `UInt`
template <std::size_t Count, class UInt>
constexpr UInt pack_dual_trits(DualTrits const* dual_trits) {
    static_assert(std::is_unsigned_v<UInt>, "UInt must be an unsigned type");

    // Required representable range: BASE^(2*Count) - 1
    constexpr unsigned digits = 2 * Count;
    constexpr unsigned long long max_needed = ipow_u(DualTrits::BASE, digits) - 1ULL;
    static_assert(max_needed <= std::numeric_limits<UInt>::max(),
                  "UInt does not have enough bits for Count dual-trits");

    UInt packed = 0;

    constexpr auto pow_base = [](size_t exponent) constexpr {
        UInt result = 1;
        for (size_t loops = 0; loops < exponent; loops++) {
            result *= DualTrits::BASE;
        }
        return result;
    };

    // Encoding order: direction first, then exponent
    UInt exponent = 1;
    for (std::size_t i = 0; i < Count; ++i) {
        packed += exponent * dual_trits[Count - 1 - i].asRawPackedBits();
        exponent *= pow_base(2);
    }

    return packed;
}


// Optional: automatically select the smallest uint type that can hold Count dual-trits
template <std::size_t Count>
using smallest_uint_for_dualtrits_t =
    std::conditional_t<(ipow_u(DualTrits::BASE, 2*Count) - 1ULL) <= std::numeric_limits<std::uint16_t>::max(), std::uint16_t,
    std::conditional_t<(ipow_u(DualTrits::BASE, 2*Count) - 1ULL) <= std::numeric_limits<std::uint32_t>::max(), std::uint32_t,
    std::uint64_t>>;

// Auto-packing API: pack_auto<Count>(ptr)
template <std::size_t Count>
constexpr smallest_uint_for_dualtrits_t<Count>
pack_auto(DualTrits const* dual_trits) {
    using U = smallest_uint_for_dualtrits_t<Count>;
    return pack_dual_trits<Count, U>(dual_trits);
}

template <std::size_t Count, class UInt>
constexpr void unpack_dual_trits(UInt packed, DualTrits* out) noexcept {
    static_assert(std::is_unsigned_v<UInt>, "UInt must be an unsigned integer type.");

    // compile-time container type capacity test
    constexpr UInt UMAX = std::numeric_limits<UInt>::max();
    constexpr bool fits = []() constexpr {
        UInt mul = 1;
        for (std::size_t i = 0; i < 2 * Count; ++i) {
            if (mul > UMAX / DualTrits::BASE) return false;
            mul *= DualTrits::BASE;
        }
        return true;
    }();
    static_assert(fits, "UInt is not wide enough for Count dual-trits (2*Count base-3 digits).");

    constexpr auto pow_base = [](size_t exponent) constexpr {
        UInt result = 1;
        for (size_t loops = 0; loops < exponent; loops++) {
            result *= DualTrits::BASE;
        }
        return result;
    };

    for (std::size_t i = 0; i < Count; ++i) {
        auto dir = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;
        auto exp = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;

        out[Count - 1 - i].setDirection(dir);
        out[Count - 1 - i].setExponent(exp);
    }
}

constexpr std::uint16_t pack5 (DualTrits const dual_trits[]) {
    return pack_dual_trits<5, std::uint16_t>(dual_trits);
}
constexpr std::uint32_t pack10(DualTrits const dual_trits[]) {
    return pack_dual_trits<10, std::uint32_t>(dual_trits);
}
constexpr std::uint64_t pack20(DualTrits const dual_trits[]) {
    return pack_dual_trits<20, std::uint64_t>(dual_trits);
}

constexpr void unpack5 (std::uint16_t packed, DualTrits* out) {
    unpack_dual_trits<5 , std::uint16_t>(packed, out);
}
constexpr void unpack10(std::uint32_t packed, DualTrits* out) {
    unpack_dual_trits<10, std::uint32_t>(packed, out);
}
constexpr void unpack20(std::uint64_t packed, DualTrits* out) {
    unpack_dual_trits<20, std::uint64_t>(packed, out);
}

#endif //PROJECT_FLOAT_PACKING_H
