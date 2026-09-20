# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required.")
endif()

function(ortgenai_replace_required file_path old_text new_text)
  if(NOT EXISTS "${file_path}")
    message(FATAL_ERROR "1DS source file not found: ${file_path}")
  endif()

  file(READ "${file_path}" contents)
  string(FIND "${contents}" "${new_text}" new_position)
  if(NOT new_position EQUAL -1)
    return()
  endif()

  string(FIND "${contents}" "${old_text}" old_position)
  if(old_position EQUAL -1)
    message(FATAL_ERROR "Expected 1DS source text was not found in ${file_path}: ${old_text}")
  endif()

  string(REPLACE "${old_text}" "${new_text}" contents "${contents}")
  file(WRITE "${file_path}" "${contents}")
endfunction()

set(fetch_curl_cmake "${SOURCE_DIR}/cmake/MatsdkFetchCurl.cmake")

ortgenai_replace_required(
  "${fetch_curl_cmake}"
  [=[      USE_SHARED_MBEDTLS_LIBRARY
      LINK_WITH_PTHREAD
      BUILD_CURL_EXE]=]
  [=[      USE_SHARED_MBEDTLS_LIBRARY
      BUILD_CURL_EXE]=])

ortgenai_replace_required(
  "${fetch_curl_cmake}"
  [=[  if(MATSDK_CURL_TLS_BACKEND_UPPER STREQUAL "MBEDTLS")
    set(USE_STATIC_MBEDTLS_LIBRARY ON)]=]
  [=[  if(MATSDK_CURL_TLS_BACKEND_UPPER STREQUAL "MBEDTLS")
    set(LINK_WITH_PTHREAD ON)
    set(USE_STATIC_MBEDTLS_LIBRARY ON)]=])

ortgenai_replace_required(
  "${fetch_curl_cmake}"
  [=[    foreach(target mbedtls mbedx509 mbedcrypto)
      matsdk_configure_fetched_static_target("${target}")
    endforeach()]=]
  [=[    foreach(target mbedtls mbedx509 mbedcrypto)
      matsdk_configure_fetched_static_target("${target}")
      target_compile_definitions("${target}" PRIVATE
        MBEDTLS_THREADING_C
        MBEDTLS_THREADING_PTHREAD)
    endforeach()]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/include/mat/config-default.h"
  [=[#define HAVE_MAT_LOGGING]=]
  [=[#if !defined(MATSDK_DISABLE_LOGGING)
#define HAVE_MAT_LOGGING
#endif]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/pal/PAL.cpp"
  [=[#if !defined(_WIN32) && defined(__linux__)
        static std::mutex m;
        static std::map<std::thread::id, pid_t> threads;]=]
  [=[#if !defined(_WIN32) && defined(__linux__)
#ifdef HAVE_MAT_LOGGING
        static std::mutex m;
        static std::map<std::thread::id, pid_t> threads;]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/pal/PAL.cpp"
  [=[        }
#else
#define     gettid()       std::this_thread::get_id()
#endif]=]
  [=[        }
#endif
#else
#define     gettid()       std::this_thread::get_id()
#endif]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/pal/TaskDispatcher_CAPI.cpp"
  [=[                catch (const std::exception& ex) {
                    LOG_ERROR("Unhandled exception in CAPI task: %s", ex.what());]=]
  [=[                catch (const std::exception& ex) {
                    (void)ex;
                    LOG_ERROR("Unhandled exception in CAPI task: %s", ex.what());]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/pal/WorkerThread.cpp"
  [=[                        catch (const std::exception& ex) {
                            LOG_ERROR("Unhandled exception in worker task: %s", ex.what());]=]
  [=[                        catch (const std::exception& ex) {
                            (void)ex;
                            LOG_ERROR("Unhandled exception in worker task: %s", ex.what());]=])

# OfflineStorageHandler::Flush() brackets its body with ILogManager::StartActivity() /
# EndActivity() using raw calls rather than a scope guard, so any exception thrown between them
# permanently leaks one pause-activity count. That is not hypothetical: during process exit,
# OnStorageRecordsSaved() -> DebugEventSource::DispatchEvent() locks a std::recursive_mutex whose
# backing static may already have been finalized, which throws std::system_error
# ("mutex lock failed: Invalid argument"). The leaked count then makes the *next* FlushAndTeardown()
# call PauseActivity() (state -> Pausing) and block forever in WaitPause(), which waits for the
# activity count to drain to zero. That is a deterministic, unrecoverable process-exit deadlock.
#
# Make the activity strictly scope-bound so the count is balanced on every exit path, including
# exceptional ones. Note this deliberately does not try to prevent the throw itself: the throwing
# lock doubles as a circuit breaker that stops DispatchEvent() from walking listener/cascaded
# containers that are themselves already destroyed. Making that mutex immortal was measured to
# remove the throw but introduce an intermittent use-after-free crash (3 failures in 75 process
# exits, versus 85/85 clean with this hunk alone). Tolerating the exception is the correct fix.
# The guard's destructor locks m_pause_mutex, a member of the still-live heap-allocated
# LogManagerImpl, so it cannot itself throw while unwinding.
#
# There is also a distinct rejected-activity path: a scheduled flush sets m_flushPending and resets
# m_flushComplete before its task starts. If teardown pauses the LogManager first, StartActivity()
# returns false; returning without signaling completion leaves WaitForFlush() blocked forever.
# Cancel the scheduled handle and complete the pending flush under m_flushLock on that path.
# Reported upstream; drop this hunk once the fix is in a pinned release.
ortgenai_replace_required(
  "${SOURCE_DIR}/lib/offline/OfflineStorageHandler.cpp"
  [=[        if (!m_logManager.StartActivity()) {
            return;
        }]=]
  [=[        if (!m_logManager.StartActivity()) {
            // Teardown can pause the LogManager after a flush has been scheduled but before its task
            // starts. Complete the rejected flush so WaitForFlush() cannot wait forever.
            LOCKGUARD(m_flushLock);
            m_flushHandle.Cancel();
            m_flushComplete.post();
            m_flushPending = false;
            return;
        }
        // Balance the activity count on every exit path, including an exception unwinding out of
        // this function; leaking it would deadlock a later FlushAndTeardown() in WaitPause().
        struct OrtGenAIActivityGuard final {
            ILogManager& log_manager;
            ~OrtGenAIActivityGuard() { log_manager.EndActivity(); }
        } ortgenai_activity_guard{m_logManager};]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/offline/OfflineStorageHandler.cpp"
  [=[        m_flushPending = false;
        m_logManager.EndActivity();]=]
  [=[        m_flushPending = false;
        // EndActivity() now runs from ortgenai_activity_guard above, on every exit path.]=])

# LogManagerProvider::Release() walks LogManagerFactory's registries to find and destroy the manager.
# The factory is a function-local static, so during static destruction it can already be destroyed by
# the time a host's exit path calls Release() -- the registries are then destroyed std::maps and
# walking them dereferences freed nodes (observed: EXC_BAD_ACCESS in LogManagerFactory::release()).
# Make the factory immortal: it is a small, fixed-size object, and leaking it keeps the registries
# readable so Release() can still perform the *real* teardown (FlushAndTeardown + worker-thread join)
# it is being asked to do. Reported upstream; drop this hunk once the fix is in a pinned release.
ortgenai_replace_required(
  "${SOURCE_DIR}/lib/api/LogManagerFactory.hpp"
  [=[        static LogManagerFactory& instance() {
            static LogManagerFactory impl;
            return impl;
        }]=]
  [=[        static LogManagerFactory& instance() {
            static LogManagerFactory& impl = *new LogManagerFactory();
            return impl;
        }]=])

# GetPAL()'s function-local static is destroyed at a point that is not ordered against the host's
# teardown call: PAL is constructed lazily on first telemetry use, so whether its static outlives the
# LogManager teardown depends on runtime timing. When it does not, LogManagerImpl::FlushAndTeardown()
# -> PAL::shutdown() releases shared_ptr members of an already-destroyed PlatformAbstractionLayer and
# faults (observed: intermittent EXC_BAD_ACCESS in ~shared_ptr<ISystemInformation>, roughly 3 in 25
# process exits). Make the PAL immortal so shutdown() always operates on live members; it is a single
# fixed-size object and PAL::shutdown() already performs the real resource teardown explicitly.
# Reported upstream; drop this hunk once the fix is in a pinned release.
ortgenai_replace_required(
  "${SOURCE_DIR}/lib/pal/PAL.cpp"
  [=[        static PlatformAbstractionLayer pal;
        return pal;]=]
  [=[        static PlatformAbstractionLayer& pal = *new PlatformAbstractionLayer();
        return pal;]=])
