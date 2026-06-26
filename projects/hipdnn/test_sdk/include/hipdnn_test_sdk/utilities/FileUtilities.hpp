// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Check if a path has the compound extension .meta.json.
///
/// std::filesystem::path::extension() returns only the last extension (".json"),
/// so "Small.meta.json" passes a naive `.json` filter. This helper detects the
/// compound extension by inspecting the stem's extension:
///   path("Small.meta.json").stem()           → "Small.meta"
///   path("Small.meta").extension()           → ".meta"
inline bool isMetaJsonFile(const std::filesystem::path& filepath)
{
    return filepath.extension() == ".json" && filepath.stem().extension() == ".meta";
}

/// Detect whether a runtime_error originates from a missing/unreadable tensor
/// data file (the loadGraphAndTensors error path). This happens when bundle
/// JSON files are present but their companion .tensor*.bin files have not been
/// pulled (e.g. DVC data unavailable). The golden reference harnesses use this
/// to GTEST_SKIP() gracefully and warn, instead of failing the test.
inline bool isTensorLoadFailure(const std::runtime_error& error)
{
    constexpr std::string_view PREFIX = "Error: could not load tensor ";
    const std::string_view      message(error.what());
    return message.size() >= PREFIX.size() && message.substr(0, PREFIX.size()) == PREFIX;
}

class ScopedDirectory
{
    std::filesystem::path _path;

public:
    ScopedDirectory(std::filesystem::path path)
    {
        if(std::filesystem::create_directory(path))
        {
            _path = std::move(path);
        }
        else
        {
            throw std::runtime_error("ScopedDirectory: Directory already exists");
        }
    }
    const std::filesystem::path& path() const
    {
        return _path;
    }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;
    ScopedDirectory(ScopedDirectory&&) = default;
    ScopedDirectory& operator=(ScopedDirectory&&) = default;
    ~ScopedDirectory()
    {
        if(!_path.empty())
        {
            std::filesystem::remove_all(_path);
        }
    }
};

/// Recursively scan a directory for bundle JSON files.
/// Returns sorted paths. Excludes meta.json. Returns empty vector on error or
/// if no bundles are found.
inline std::vector<std::filesystem::path>
    scanBundleJsonFiles(const std::filesystem::path& directory)
{
    std::vector<std::filesystem::path> paths;
    try
    {
        for(const auto& entry : std::filesystem::recursive_directory_iterator(directory))
        {
            if(entry.path().extension() == ".json" && entry.path().filename() != "meta.json"
               && !isMetaJsonFile(entry.path()))
            {
                paths.push_back(entry.path());
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Warning: failed to scan golden reference data in " << directory << ": "
                  << e.what() << '\n';
        return {};
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

/// Scan integration test bundles directory for bundle JSON files.
/// Returns a gtest parameter generator. On failure or empty directory, returns
/// a sentinel empty-path so SetUp() can GTEST_SKIP() gracefully.
inline auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    auto dir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
               / "../lib/integration_test_bundles" / subDirectory;

    auto paths = scanBundleJsonFiles(dir);
    if(paths.empty())
    {
        return testing::ValuesIn(std::vector<std::filesystem::path>{""});
    }
    return testing::ValuesIn(paths);
}

}
