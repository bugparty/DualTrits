#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

/*
 * Extract X bits from a byte array and pack them into a single byte.
 * 
 * This function extracts (1 + EXPONENT + MANTISSA) bits from the source byte array
 * at the specified byte and bit offsets, and returns them in the high bits of a byte.
 * Used for extracting FPx values from packed storage.
 *
 * Template Parameters:
 *   EXPONENT - Number of exponent bits in the floating point format
 *   MANTISSA - Number of mantissa bits in the floating point format
 *
 * Parameters:
 *   Bytes      - Pointer to the source byte array containing packed FPx values
 *   ByteOffset - Byte offset from the start of Bytes array to begin extraction
 *   BitOffset  - Bit offset within the starting bytes (0-7) for fine-grained positioning
 *
 * Returns:
 *   A byte containing the extracted bits (sign + exponent + mantissa) in the high bits
 */
template<int EXPONENT, int MANTISSA>
unsigned char Extract_X_Bits_To_A_Byte(unsigned char* Bytes, int ByteOffset, int BitOffset){
    static_assert(sizeof(unsigned int)==4, "unsigned int must be 4 bytes");
    unsigned int tmp_int32_word=0;
    unsigned char* uchar_ptr = reinterpret_cast<unsigned char*>(&tmp_int32_word);
    uchar_ptr[3] = Bytes[ByteOffset+0];
    uchar_ptr[2] = Bytes[ByteOffset+1];
    tmp_int32_word = tmp_int32_word << BitOffset;
    //
    signed int mask = 0x80000000;
    mask = mask >> (EXPONENT+MANTISSA);
    tmp_int32_word &= mask;
    //
    unsigned char out = uchar_ptr[3];
    return out;
}

#endif