/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP

#include "unit_conv_solver.hpp"

#include <miopen/execution_context.hpp>

#include <string_view>

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // YOLO26x depthwise shapes.
        TestCase{{1, 384, 80, 80}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        TestCase{{1, 768, 40, 40}, {768, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 768, datatype},
        TestCase{{1, 384, 20, 20}, {384, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 384, datatype},
        // Strided, unpadded, larger filter and odd-sized variants. The 33x17
        // case yields out_h = 17, a prime, so the row block cannot divide the
        // output height and the last row block runs short.
        TestCase{{2, 128, 33, 17}, {128, 1, 3, 3}, {1, 1}, {2, 2}, {1, 1}, 128, datatype},
        TestCase{{1, 96, 41, 41}, {96, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 96, datatype},
        TestCase{{1, 64, 56, 56}, {64, 1, 5, 5}, {2, 2}, {1, 1}, {1, 1}, 64, datatype},
        // Dilated. Stride 1 is the atrous case, where no input row survives to
        // the next output row and the window is reloaded per row; stride 2 is
        // the case where rows are reused across the slide.
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {1, 1}, {2, 2}, 64, datatype},
        TestCase{{1, 64, 28, 28}, {64, 1, 3, 3}, {2, 2}, {2, 2}, {2, 2}, 64, datatype},
        // Asymmetric filter, padding, stride and dilation, so that swapping a
        // height constant for a width one cannot pass unnoticed.
        TestCase{{1, 64, 32, 32}, {64, 1, 5, 3}, {2, 1}, {1, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {1, 1}, {2, 1}, {1, 1}, 64, datatype},
        TestCase{{1, 64, 32, 32}, {64, 1, 3, 3}, {2, 1}, {1, 1}, {2, 1}, 64, datatype},
        // Largest register window the solver admits: 7x7 needs 105 of the 128
        // float slots kMaxWindowRegs allows.
        TestCase{{1, 64, 24, 24}, {64, 1, 7, 7}, {3, 3}, {1, 1}, {1, 1}, 64, datatype},
        // A single output pixel, over an input narrower than one lane's column
        // span, so every row load takes the bounds-checked path.
        TestCase{{1, 64, 3, 3}, {64, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, 64, datatype},
        // clang-format on
    };
}

// A shape squarely inside the solver's declared scope, used as the control for
// the rejection cases below.
miopen::unit_tests::ConvTestCase GetSupportedCase()
{
    return miopen::unit_tests::ConvTestCase{
        {1, 64, 28, 28}, {64, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 64, miopenHalf};
}

// Ask the solver directly, against a mock device, without needing that device
// to be present.
bool IsApplicableOn(std::string_view dev_name,
                    const miopen::unit_tests::ConvTestCase& conv_config,
                    miopen::conv::Direction direction)
{
    const auto problem = conv_config.GetProblemDescription(direction);
    auto handle        = MockHandle{DevDescription{dev_name, 40, 32}, false};
    auto ctx           = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);
    problem.SetupComputeType(ctx);
    return miopen::solver::conv::ConvDirectDepthwiseFwd2D{}.IsApplicable(ctx, problem);
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx110X | Gpu::gfx115X |
                                                              Gpu::gfx120X);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16  = GPU_UnitTestConvSolverFwd_FP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16 = GPU_UnitTestConvSolverFwd_BFP16;
using GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32  = GPU_UnitTestConvSolverFwd_FP32;

using CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityFwd_NONE;

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32, ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

TEST_P(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
       ConvDirectDepthwiseFwd2D)
{
    this->RunTest(miopen::solver::conv::ConvDirectDepthwiseFwd2D{});
}

// The kernel's register footprint and slot arithmetic are only valid inside the
// bounds IsApplicable enforces, so widening any of them silently is a bug. Pin
// each rejection.
TEST(CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DApplicability_NONE, ApplicabilityBounds)
{
    using TestCase     = miopen::unit_tests::ConvTestCase;
    constexpr auto fwd = miopen::conv::Direction::Forward;

    // Control: without this the rejections below prove nothing.
    EXPECT_TRUE(IsApplicableOn("gfx1151", GetSupportedCase(), fwd));

    // gfx1250 shares the "gfx12" prefix with gfx120X but is not claimed.
    EXPECT_FALSE(IsApplicableOn("gfx1250", GetSupportedCase(), fwd));
    // CDNA reaches this shape class through the grouped XDLOPS path.
    EXPECT_FALSE(IsApplicableOn("gfx942", GetSupportedCase(), fwd));

    // Forward only.
    EXPECT_FALSE(
        IsApplicableOn("gfx1151", GetSupportedCase(), miopen::conv::Direction::BackwardData));

    // Below kMinChannels the grouped tile is not yet degenerate.
    EXPECT_FALSE(IsApplicableOn(
        "gfx1151",
        TestCase{{1, 16, 28, 28}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 16, miopenHalf},
        fwd));

    // More than one channel per group is not depthwise.
    EXPECT_FALSE(IsApplicableOn(
        "gfx1151",
        TestCase{{1, 64, 28, 28}, {64, 2, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, miopenHalf},
        fwd));

    // Filter extent past kMaxFilterExtent.
    EXPECT_FALSE(IsApplicableOn(
        "gfx1151",
        TestCase{{1, 64, 32, 32}, {64, 1, 9, 9}, {4, 4}, {1, 1}, {1, 1}, 64, miopenHalf},
        fwd));

    // Filter extent is fine, but dilation stretches the register window past
    // kMaxWindowRegs (13 * 14 + 49 = 231 slots).
    EXPECT_FALSE(IsApplicableOn(
        "gfx1151",
        TestCase{{1, 64, 32, 32}, {64, 1, 7, 7}, {6, 6}, {1, 1}, {2, 2}, 64, miopenHalf},
        fwd));
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverConvDirectDepthwiseFwd2D_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverConvDirectDepthwiseFwd2DDevApplicabilityFwd_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenHalf)[0])));

#endif // MIOPEN_BACKEND_HIP
