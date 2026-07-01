/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#ifdef MIOPEN_HIP_RUNTIME_COMPILE
// During HIP run-time compilation (HIPRTC), the fixed-width integer types are
// provided here instead of pulling in the host <cstdint>. However, some system
// or HIP headers that get transitively included during HIPRTC compilation may
// already provide the standard <stdint.h> typedefs. On many Linux systems glibc
// defines uint64_t as 'unsigned long' whereas __hip_internal::uint64_t is
// 'unsigned long long' (and int64_t as 'long' vs 'long long'). Re-declaring an
// already-defined typedef with a different underlying type is a hard error that
// surfaces as HIPRTC_ERROR_COMPILATION -> RuntimeError: miopenStatusUnknownError
// at run time (see ROCm/TheRock#6258). Guard against this by only emitting our
// definitions when the standard <stdint.h> header has not already been included.
#if !defined(_STDINT_H) && !defined(_STDINT_H_) && !defined(_GCC_WRAP_STDINT_H) && \
    !defined(_STDINT)
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
#if HIP_PACKAGE_VERSION_FLAT >= 6000025000ULL
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef __hip_internal::uint64_t uint64_t;
typedef __hip_internal::int64_t int64_t;
#endif
#endif // stdint.h not already included

#else
#include <cstdint> // int8_t, int16_t
#endif
