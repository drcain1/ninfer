# Building and running NInfer on Windows

This guide covers native Windows 11 x64 source builds for the same NVIDIA GeForce RTX 5090
(`sm_120a`) target as Linux. Windows uses the same `.ninfer` artifacts, CLI behavior, and HTTP APIs.

## Requirements

- Windows 11 x64;
- NVIDIA GeForce RTX 5090 with a driver supporting CUDA 13.1;
- CUDA Toolkit 13.1 or newer;
- Visual Studio 2022 or newer with **Desktop development with C++**;
- CMake 3.28 or newer;
- [vcpkg](https://github.com/microsoft/vcpkg).

The build rejects CUDA architectures other than `120a`. FFmpeg and libcurl are installed from the
repository's pinned `vcpkg.json` manifest during configuration.

## Install vcpkg

From PowerShell:

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
```

## Build

Run these commands from an **x64 Native Tools Command Prompt for Visual Studio**, or another shell
where the MSVC and CUDA toolchains are available. The example selects Visual Studio 2022; use the
generator installed on your machine when building with a newer release:

```powershell
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-windows --config Release --parallel
```

The default build produces:

```text
build-windows/apps/Release/ninfer.exe
build-windows/apps/Release/ninfer-serve.exe
```

Tests and benchmarks remain opt-in. Add `-DBUILD_TESTING=ON` at configure time and run:

```powershell
cmake --build build-windows --config Release --parallel
ctest --test-dir build-windows -C Release --output-on-failure
```

## Run

After downloading a model artifact as described in the project README:

```powershell
.\build-windows\apps\Release\ninfer.exe models\qwen3_6_27b.ninfer `
  --prompt "Explain prefill and decode in three sentences." `
  --max-context 16384 --max-new 256
```

To run the HTTP server:

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_6_27b.ninfer `
  --max-context 16384 --kv-capacity auto --max-concurrency 2
```

The API is available at `http://127.0.0.1:8080/v1`.

The CUDA runtime is statically linked. FFmpeg, libcurl, zlib, and their runtime DLLs are supplied by
the vcpkg build tree; keep the vcpkg-installed runtime DLLs beside the executables when packaging or
moving them outside the build tree.
