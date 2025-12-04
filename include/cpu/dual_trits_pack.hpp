//
// Created by bowman on 11/8/25.
//

#ifndef PROJECT_FLOAT_PACKING_H
#define PROJECT_FLOAT_PACKING_H
#include <cstdint>
#include "common/DualTrits.hpp"
#include <limits>
#include <type_traits>
#include <vector>

namespace details {

template <std::size_t TritsPerPack, class UInt>
constexpr bool fits() {
    constexpr UInt UMAX = std::numeric_limits<UInt>::max();
    UInt mul = 1;
    for (std::size_t i = 0; i < 2 * TritsPerPack; ++i) {
        if (mul > UMAX / DualTrits::BASE) return false;
        mul *= DualTrits::BASE;
    }
    return true;
}

}


// Simple constexpr integer power
constexpr unsigned long long ipow_u(unsigned base, unsigned exp) {
    unsigned long long r = 1;
    while (exp--) r *= base;
    return r;
}

// Generic packer: pack `TritsPerPack` DualTrits into unsigned integer type `UInt`
template <std::size_t TritsPerPack, class UInt>
constexpr UInt pack_dual_trits(DualTrits const* dual_trits) {
#if __cplusplus >= 202002L
    static_assert(std::is_unsigned_v<UInt>, "UInt must be an unsigned type");
#else
    static_assert(std::is_unsigned<UInt>::value, "UInt must be an unsigned integer type.");
#endif

    // Required representable range: BASE^(2*TritsPerPack) - 1
    constexpr unsigned digits = 2 * TritsPerPack;
    constexpr unsigned long long max_needed = ipow_u(DualTrits::BASE, digits) - 1ULL;
    static_assert(max_needed <= std::numeric_limits<UInt>::max(),
                  "UInt does not have enough bits for TritsPerPack dual-trits");

    UInt packed = 0;
    UInt multiplier = 1;

    // Encoding order: direction first, then exponent
    for (std::size_t i = 0; i < TritsPerPack; ++i) {
        const DualTrits& t = dual_trits[TritsPerPack - 1 - i];

        packed += static_cast<UInt>(t.getDirection()) * multiplier;
        multiplier *= DualTrits::BASE;

        packed += static_cast<UInt>(t.getExponent()) * multiplier;
        multiplier *= DualTrits::BASE;
    }

    return packed;
}

template <std::size_t TritsPerPack, class UInt>
std::vector<UInt> batch_pack_dual_trits(DualTrits const dual_trits[], size_t n) {
    size_t totalPacks = (n + TritsPerPack - 1) / TritsPerPack;
    std::vector<UInt> packed(totalPacks);

    size_t offset = n % TritsPerPack;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (offset != 0) {
                std::vector<DualTrits> firstPackVector(TritsPerPack);
                DualTrits* firstPack = firstPackVector.data();
                size_t dualTritsIndex = 0;
                for (size_t firstPackIndex = TritsPerPack - offset; firstPackIndex < TritsPerPack; firstPackIndex++) {
                    firstPack[firstPackIndex] = dual_trits[dualTritsIndex];
                    dualTritsIndex++;
                }
                packed[0] = pack_dual_trits<TritsPerPack, UInt>(firstPack);
            } else {
                packed[0] = pack_dual_trits<TritsPerPack, UInt>(dual_trits);
            }
        }

        #pragma omp section
        {
            #pragma omp parallel for schedule(static, TritsPerPack)
            for (size_t packIndex = 1; packIndex < totalPacks; packIndex++) {
                packed[packIndex] = pack_dual_trits<TritsPerPack, UInt>(offset + ((packIndex - 1) * TritsPerPack) + dual_trits);
            }
        }
    }
    return packed;
}


// Optional: automatically select the smallest uint type that can hold TritsPerPack dual-trits
template <std::size_t TritsPerPack>
using smallest_uint_for_dualtrits_t =
    std::conditional_t<(ipow_u(DualTrits::BASE, 2*TritsPerPack) - 1ULL) <= std::numeric_limits<std::uint16_t>::max(), std::uint16_t,
    std::conditional_t<(ipow_u(DualTrits::BASE, 2*TritsPerPack) - 1ULL) <= std::numeric_limits<std::uint32_t>::max(), std::uint32_t,
    std::uint64_t>>;

// Auto-packing API: pack_auto<TritsPerPack>(ptr)
template <std::size_t TritsPerPack>
constexpr smallest_uint_for_dualtrits_t<TritsPerPack>
pack_auto(DualTrits const* dual_trits) {
    using U = smallest_uint_for_dualtrits_t<TritsPerPack>;
    return batch_pack_dual_trits<TritsPerPack, U>(dual_trits);
}

template <std::size_t TritsPerPack, class UInt>
constexpr void unpack_dual_trits(UInt packed, DualTrits* out) noexcept {
#if __cplusplus >= 202002L
    static_assert(std::is_unsigned_v<UInt>, "UInt must be an unsigned integer type.");
#else
    static_assert(std::is_unsigned<UInt>::value, "UInt must be an unsigned integer type.");
#endif

    // compile-time container type capacity test
    constexpr bool fits = details::fits<TritsPerPack, UInt>();
    static_assert(fits, "UInt is not wide enough for Count dual-trits (2*Count base-3 digits).");

    for (std::size_t i = 0; i < TritsPerPack; ++i) {
        auto dir = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;
        auto exp = static_cast<std::uint16_t>(packed % DualTrits::BASE);
        packed /= DualTrits::BASE;

        out[TritsPerPack - 1 - i].setDirection(dir);
        out[TritsPerPack - 1 - i].setExponent(exp);
    }
}

// unpack_dual_trits assumes that for n packed values, there will be n * Count
// elements allocated for DualTrits in out.
template <std::size_t TritsPerPack, class UInt>
void unpack_dual_trits(UInt* packed, DualTrits* out, size_t n) noexcept {
    #pragma omp parallel for schedule(static, Count)
    for (size_t i = 0; i < n; i++) {
        unpack_dual_trits<TritsPerPack, UInt>(packed[i], out + (TritsPerPack * i));
    }
}

constexpr std::uint16_t pack5(DualTrits const dual_trits[]) {
    return pack_dual_trits<5, std::uint16_t>(dual_trits);
}
inline std::vector<std::uint16_t> batch_pack5(DualTrits const dual_trits[], size_t n) {
    return batch_pack_dual_trits<5, std::uint16_t>(dual_trits, n);
}
constexpr std::uint32_t pack10(DualTrits const dual_trits[]) {
    return pack_dual_trits<10, std::uint32_t>(dual_trits);
}
inline std::vector<std::uint32_t> batch_pack10(DualTrits const dual_trits[], size_t n) {
    return batch_pack_dual_trits<10, std::uint32_t>(dual_trits, n);
}
constexpr std::uint64_t pack20(DualTrits const dual_trits[]) {
    return pack_dual_trits<20, std::uint64_t>(dual_trits);
}
inline std::vector<std::uint64_t> batch_pack20(DualTrits const dual_trits[], size_t n) {
    return batch_pack_dual_trits<20, std::uint64_t>(dual_trits, n);
}

inline void unpack5(std::uint16_t packed, DualTrits* out) {
    unpack_dual_trits<5, std::uint16_t>(packed, out);
}
inline void batch_unpack5(std::uint16_t* packed, DualTrits* out, size_t n) {
    unpack_dual_trits<5, std::uint16_t>(packed, out, n);
}
inline void unpack10(std::uint32_t packed, DualTrits* out) {
    unpack_dual_trits<10, std::uint32_t>(packed, out);
}
inline void batch_unpack10(std::uint32_t* packed, DualTrits* out, size_t n) {
    unpack_dual_trits<10, std::uint32_t>(packed, out, n);
}
inline void unpack20(std::uint64_t packed, DualTrits* out) {
    unpack_dual_trits<20, std::uint64_t>(packed, out);
}
inline void batch_unpack20(std::uint64_t* packed, DualTrits* out, size_t n) {
    unpack_dual_trits<20, std::uint64_t>(packed, out, n);
}

#endif //PROJECT_FLOAT_PACKING_H
