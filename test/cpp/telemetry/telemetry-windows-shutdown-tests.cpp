// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s (Win32 error: %lu)\n", message, GetLastError());
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

int wmain(int argc, wchar_t* argv[]) {
  if (argc == 5 && std::wstring_view{argv[1]} == L"--child") {
    const std::wstring_view mode{argv[2]};
    const HMODULE library = LoadLibraryW(argv[3]);
    Check(library != nullptr, "cannot load telemetry fixture");
    using Run = void (*)(const wchar_t*, HANDLE);
    const auto run = reinterpret_cast<Run>(GetProcAddress(library, "GenAiTelemetryTestLifecycle"));
    Check(run != nullptr, "cannot resolve lifecycle fixture");
    wchar_t* end = nullptr;
    const auto handle = std::wcstoull(argv[4], &end, 10);
    Check(end != argv[4] && *end == L'\0', "invalid inherited event");
    run(argv[2], reinterpret_cast<HANDLE>(static_cast<uintptr_t>(handle)));
    if (mode == L"process-exit" || mode == L"locked-process-exit" ||
        mode == L"shutdown-locked-process-exit") ExitProcess(0);
    Check(FreeLibrary(library) != FALSE, "cannot unload telemetry fixture");
    return 0;
  }

  Check(argc == 3, "expected lifecycle mode and fixture DLL");
  const std::wstring_view mode{argv[1]};
  Check(mode == L"process-exit" || mode == L"locked-process-exit" || mode == L"shutdown-locked-process-exit" ||
            mode == L"normal-destruction" || mode == L"explicit-shutdown" || mode == L"explicit-unload",
        "unknown lifecycle mode");
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  const HANDLE verified = CreateEventW(&attributes, TRUE, FALSE, nullptr);
  Check(verified != nullptr, "cannot create inherited assertion event");
  wchar_t executable[32768];
  const DWORD length = GetModuleFileNameW(nullptr, executable, ARRAYSIZE(executable));
  Check(length != 0 && length < ARRAYSIZE(executable), "cannot resolve child executable");
  std::wstring command = L"\"" + std::wstring{executable} + L"\" --child " + std::wstring{mode} +
                         L" \"" + argv[2] + L"\" " + std::to_wstring(reinterpret_cast<uintptr_t>(verified));
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION child{};
  Check(CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child) != FALSE,
        "cannot start lifecycle child");
  CloseHandle(child.hThread);
  const DWORD wait = WaitForSingleObject(child.hProcess, 10000);
  if (wait != WAIT_OBJECT_0) {
    TerminateProcess(child.hProcess, EXIT_FAILURE);
    WaitForSingleObject(child.hProcess, 5000);
  }
  DWORD code = EXIT_FAILURE;
  const BOOL exited = GetExitCodeProcess(child.hProcess, &code);
  CloseHandle(child.hProcess);
  const bool assertions_ran = WaitForSingleObject(verified, 0) == WAIT_OBJECT_0;
  CloseHandle(verified);
  Check(wait == WAIT_OBJECT_0 && exited && code == 0, "lifecycle child failed or timed out");
  Check(assertions_ran, "child exited without verifying telemetry destruction");
  std::fputs("PASS: DLL telemetry lifecycle verified\n", stderr);
  return 0;
}
