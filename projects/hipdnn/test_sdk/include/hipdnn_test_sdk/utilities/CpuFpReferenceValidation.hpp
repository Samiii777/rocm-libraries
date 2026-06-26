// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

template <class T>
class CpuFpReferenceValidation : public IReferenceValidation
{
public:
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    CpuFpReferenceValidation(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                             // NOLINTNEXTLINE(readability-redundant-casting)
                             float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
        : _absoluteTolerance(absoluteTolerance)
        , _relativeTolerance(relativeTolerance)
    {
        if(absoluteTolerance < 0.0f || relativeTolerance < 0.0f || std::isnan(absoluteTolerance)
           || std::isnan(relativeTolerance) || std::isinf(absoluteTolerance)
           || std::isinf(relativeTolerance))
        {
            throw std::invalid_argument("Tolerances must be finite and non-negative");
        }
    }

    ~CpuFpReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        std::atomic<bool> result(true);

        // Diagnostics: track the single worst (largest absolute-difference) failing element, the
        // maximum relative difference, and the total number of failing elements. This converts an
        // otherwise opaque "Mismatch found" assertion into an actionable report (max abs diff, max
        // rel diff, worst index, expected vs actual, failing count), which is essential for
        // triaging flaky/marginal numerical mismatches.
        // See https://github.com/ROCm/rocm-libraries/issues/8638.
        // The worst-element bookkeeping is serialized behind a mutex but is only entered when an
        // element has already failed the tolerance check, so it does not affect the fast path.
        std::mutex worstMutex;
        std::atomic<int64_t> failingCount(0);
        bool haveWorst = false;
        float worstAbsDiff = 0.0f;
        float worstRefValue = 0.0f;
        float worstImplValue = 0.0f;
        float worstThreshold = 0.0f;
        std::vector<int64_t> worstIndices;
        // Track the maximum relative difference independently of the maximum absolute difference,
        // since the worst element by either metric may differ.
        float maxRelDiff = 0.0f;

        auto recordWorst = [&](const std::vector<int64_t>& indices,
                               float refValueF,
                               float implValueF,
                               float absDiff,
                               float threshold) {
            const std::lock_guard<std::mutex> lock(worstMutex);
            const float denom = std::fabs(refValueF);
            const float relDiff = denom > 0.0f ? absDiff / denom : absDiff;
            maxRelDiff = std::max(maxRelDiff, relDiff);
            if(!haveWorst || absDiff > worstAbsDiff)
            {
                haveWorst = true;
                worstAbsDiff = absDiff;
                worstRefValue = refValueF;
                worstImplValue = implValueF;
                worstThreshold = threshold;
                worstIndices = indices;
            }
        };

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            using hipdnn_data_sdk::types::fabs;
            using hipdnn_data_sdk::types::isnan;
            using hipdnn_data_sdk::types::isinf;
            T refValue = refView.getHostValue(indices);
            T implValue = implView.getHostValue(indices);

            if(isnan(refValue) || isinf(refValue) || isnan(implValue) || isinf(implValue))
            {
                HIPDNN_SDK_LOG_ERROR(
                    "NaN or Inf detected at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ". This may indicate an output element was not written by the operation.");
                failingCount.fetch_add(1, std::memory_order_relaxed);
                result.store(false, std::memory_order_relaxed);
                return result.load(std::memory_order_relaxed);
            }

            auto absDiff = fabs(static_cast<float>(implValue) - static_cast<float>(refValue));
            auto threshold
                = _absoluteTolerance + _relativeTolerance * fabs(static_cast<float>(refValue));

            if(absDiff > threshold)
            {
                // Log error and mark as failed
                HIPDNN_SDK_LOG_ERROR(
                    "Validation failed at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ", absolute difference = " << absDiff << ", threshold = " << threshold
                    << ", difference - threshold = " << (absDiff - threshold)
                    << ", (atol=" << _absoluteTolerance << ", rtol=" << _relativeTolerance << ")");
                failingCount.fetch_add(1, std::memory_order_relaxed);
                recordWorst(indices,
                            static_cast<float>(refValue),
                            static_cast<float>(implValue),
                            static_cast<float>(absDiff),
                            static_cast<float>(threshold));
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        // Emit a single, concise diagnostics summary on failure so flaky/marginal mismatches are
        // immediately actionable from the test log without re-running with extra instrumentation.
        if(haveWorst)
        {
            const float worstRelDiff = std::fabs(worstRefValue) > 0.0f
                                           ? worstAbsDiff / std::fabs(worstRefValue)
                                           : worstAbsDiff;
            HIPDNN_SDK_LOG_ERROR(
                "allClose summary: "
                << failingCount.load(std::memory_order_relaxed) << " of " << reference.elementCount()
                << " elements failed. Worst element at indices " << StreamVec(worstIndices)
                << ": expected (reference) = " << worstRefValue
                << ", actual (implementation) = " << worstImplValue
                << ", max absolute difference = " << worstAbsDiff
                << " (threshold = " << worstThreshold << ", exceeded by "
                << (worstAbsDiff - worstThreshold)
                << "), relative difference at worst element = " << worstRelDiff
                << ", max relative difference (any element) = " << maxRelDiff
                << ", (atol=" << _absoluteTolerance << ", rtol=" << _relativeTolerance << ")");
        }

        return result.load();
    }

private:
    float _absoluteTolerance;
    float _relativeTolerance;
};

template <class T>
class CpuIntReferenceValidation : public IReferenceValidation
{
public:
    CpuIntReferenceValidation() = default;
    ~CpuIntReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        std::atomic<bool> result(true);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            T refValue = refView.getHostValue(indices);
            T implValue = implView.getHostValue(indices);

            if(refValue == std::numeric_limits<T>::max()
               || implValue == std::numeric_limits<T>::max())
            {
                HIPDNN_SDK_LOG_ERROR(
                    "Sentinel value detected at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ". This may indicate an output element was not written by the operation.");
                result.store(false, std::memory_order_relaxed);
                return result.load(std::memory_order_relaxed);
            }

            T absDiff = static_cast<T>(hipdnn_data_sdk::types::abs(implValue - refValue));

            // Integer values must be equal
            if(absDiff > 0)
            {
                // Log error and mark as failed
                HIPDNN_SDK_LOG_ERROR("Validation failed for integer values at indices "
                                     << StreamVec(indices) << ": reference value = " << refValue
                                     << ", implementation value = " << implValue
                                     << ", absolute difference = " << absDiff);
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return result.load();
    }
};

inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                            float absoluteTolerance = std::numeric_limits<float>::epsilon(),
                            float relativeTolerance = std::numeric_limits<float>::epsilon())
{
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<CpuFpReferenceValidation<float>>(absoluteTolerance,
                                                                 relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return std::make_unique<CpuFpReferenceValidation<hipdnn_data_sdk::types::half>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<CpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<CpuFpReferenceValidation<double>>(absoluteTolerance,
                                                                  relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
        return std::make_unique<CpuIntReferenceValidation<int8_t>>();
    case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
        return std::make_unique<CpuIntReferenceValidation<uint8_t>>();
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return std::make_unique<CpuIntReferenceValidation<int32_t>>();
    default:
        throw std::runtime_error("Unsupported data type for allClose validator");
    }
}

template <typename T>
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                            float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
{
    if constexpr(std::is_integral_v<T>)
    {
        return std::make_unique<CpuIntReferenceValidation<T>>();
    }
    else
    {
        return std::make_unique<CpuFpReferenceValidation<T>>(absoluteTolerance, relativeTolerance);
    }
}

} // namespace hipdnn_test_sdk::utilities
