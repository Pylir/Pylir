//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Pages.hpp"

#include <pylir/Support/Macros.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

static std::size_t pageSize;

void __attribute__((constructor(200))) pageSizeInit() {
#ifdef _WIN32
  SYSTEM_INFO systemInfo;
  GetSystemInfo(&systemInfo);
  // This is technically not the page size, but it is granularity VirtualAlloc
  // works with.
  pageSize = systemInfo.dwAllocationGranularity;
#else
  pageSize = sysconf(_SC_PAGESIZE);
#endif
}

std::size_t pylir::rt::getPageSize() {
  return pageSize;
}

void pylir::rt::PageDeleter::operator()(std::byte* page) const noexcept {
#ifdef _WIN32
  VirtualFree(page, size, MEM_RELEASE);
#else
  munmap(page, size);
#endif
}

pylir::rt::PagePtr pylir::rt::pageAlloc(std::size_t pageCount) {
  PYLIR_ASSERT(pageCount > 0);
  std::size_t bytes = getPageSize() * pageCount;
#ifdef _WIN32
  return {reinterpret_cast<std::byte*>(VirtualAlloc(
              nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)),
          bytes};
#else
  return {
      reinterpret_cast<std::byte*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE, 0, 0)),
      bytes};
#endif
}
