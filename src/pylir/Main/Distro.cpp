//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Distro.hpp"

//===--- Distro.cpp - Linux distribution detection support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Threading.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

static pylir::Distro::DistroType detectOsRelease() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile("/etc/os-release");
  if (!file)
    file = llvm::MemoryBuffer::getFile("/usr/lib/os-release");

  if (!file)
    return pylir::Distro::UnknownDistro;

  llvm::SmallVector<llvm::StringRef, 16> lines;
  file.get()->getBuffer().split(lines, "\n");
  pylir::Distro::DistroType version = pylir::Distro::UnknownDistro;

  // Obviously this can be improved a lot.
  for (llvm::StringRef line : lines) {
    if (version != pylir::Distro::UnknownDistro || !line.starts_with("ID="))
      continue;

    version = llvm::StringSwitch<pylir::Distro::DistroType>(line.substr(3))
                  .Case("alpine", pylir::Distro::AlpineLinux)
                  .Case("fedora", pylir::Distro::Fedora)
                  .Case("gentoo", pylir::Distro::Gentoo)
                  .Case("arch", pylir::Distro::ArchLinux)
                  // On SLES, /etc/os-release was introduced in SLES 11.
                  .Case("sles", pylir::Distro::OpenSUSE)
                  .Case("opensuse", pylir::Distro::OpenSUSE)
                  .Case("exherbo", pylir::Distro::Exherbo)
                  .Default(pylir::Distro::UnknownDistro);
  }
  return version;
}

static pylir::Distro::DistroType detectLsbRelease() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile("/etc/lsb-release");
  if (!file)
    return pylir::Distro::UnknownDistro;

  llvm::SmallVector<llvm::StringRef, 16> lines;
  file.get()->getBuffer().split(lines, "\n");
  pylir::Distro::DistroType version = pylir::Distro::UnknownDistro;

  for (llvm::StringRef line : lines) {
    if (version != pylir::Distro::UnknownDistro ||
        !line.starts_with("DISTRIB_CODENAME="))
      continue;
    version = llvm::StringSwitch<pylir::Distro::DistroType>(line.substr(17))
                  .Case("hardy", pylir::Distro::UbuntuHardy)
                  .Case("intrepid", pylir::Distro::UbuntuIntrepid)
                  .Case("jaunty", pylir::Distro::UbuntuJaunty)
                  .Case("karmic", pylir::Distro::UbuntuKarmic)
                  .Case("lucid", pylir::Distro::UbuntuLucid)
                  .Case("maverick", pylir::Distro::UbuntuMaverick)
                  .Case("natty", pylir::Distro::UbuntuNatty)
                  .Case("oneiric", pylir::Distro::UbuntuOneiric)
                  .Case("precise", pylir::Distro::UbuntuPrecise)
                  .Case("quantal", pylir::Distro::UbuntuQuantal)
                  .Case("raring", pylir::Distro::UbuntuRaring)
                  .Case("saucy", pylir::Distro::UbuntuSaucy)
                  .Case("trusty", pylir::Distro::UbuntuTrusty)
                  .Case("utopic", pylir::Distro::UbuntuUtopic)
                  .Case("vivid", pylir::Distro::UbuntuVivid)
                  .Case("wily", pylir::Distro::UbuntuWily)
                  .Case("xenial", pylir::Distro::UbuntuXenial)
                  .Case("yakkety", pylir::Distro::UbuntuYakkety)
                  .Case("zesty", pylir::Distro::UbuntuZesty)
                  .Case("artful", pylir::Distro::UbuntuArtful)
                  .Case("bionic", pylir::Distro::UbuntuBionic)
                  .Case("cosmic", pylir::Distro::UbuntuCosmic)
                  .Case("disco", pylir::Distro::UbuntuDisco)
                  .Case("eoan", pylir::Distro::UbuntuEoan)
                  .Case("focal", pylir::Distro::UbuntuFocal)
                  .Case("groovy", pylir::Distro::UbuntuGroovy)
                  .Case("hirsute", pylir::Distro::UbuntuHirsute)
                  .Case("impish", pylir::Distro::UbuntuImpish)
                  .Case("jammy", pylir::Distro::UbuntuJammy)
                  .Default(pylir::Distro::UnknownDistro);
  }
  return version;
}

static pylir::Distro::DistroType detectDistro() {
  pylir::Distro::DistroType version;

  // Newer freedesktop.org's compilant systemd-based systems
  // should provide /etc/os-release or /usr/lib/os-release.
  version = detectOsRelease();
  if (version != pylir::Distro::UnknownDistro)
    return version;

  // Older systems might provide /etc/lsb-release.
  version = detectLsbRelease();
  if (version != pylir::Distro::UnknownDistro)
    return version;

  // Otherwise try some distro-specific quirks for RedHat...
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile("/etc/redhat-release");

  if (file) {
    llvm::StringRef data = file.get()->getBuffer();
    if (data.starts_with("Fedora release"))
      return pylir::Distro::Fedora;

    if (data.starts_with("Red Hat Enterprise Linux") ||
        data.starts_with("CentOS") || data.starts_with("Scientific Linux")) {
      if (data.contains("release 7"))
        return pylir::Distro::RHEL7;

      if (data.contains("release 6"))
        return pylir::Distro::RHEL6;

      if (data.contains("release 5"))
        return pylir::Distro::RHEL5;
    }
    return pylir::Distro::UnknownDistro;
  }

  // ...for Debian
  file = llvm::MemoryBuffer::getFile("/etc/debian_version");
  if (file) {
    llvm::StringRef data = file.get()->getBuffer();
    // Contents: < major.minor > or < codename/sid >
    int majorVersion;
    if (!data.split('.').first.getAsInteger(10, majorVersion)) {
      switch (majorVersion) {
      case 5: return pylir::Distro::DebianLenny;
      case 6: return pylir::Distro::DebianSqueeze;
      case 7: return pylir::Distro::DebianWheezy;
      case 8: return pylir::Distro::DebianJessie;
      case 9: return pylir::Distro::DebianStretch;
      case 10: return pylir::Distro::DebianBuster;
      case 11: return pylir::Distro::DebianBullseye;
      case 12: return pylir::Distro::DebianBookworm;
      default: return pylir::Distro::UnknownDistro;
      }
    }
    return llvm::StringSwitch<pylir::Distro::DistroType>(data.split("\n").first)
        .Case("squeeze/sid", pylir::Distro::DebianSqueeze)
        .Case("wheezy/sid", pylir::Distro::DebianWheezy)
        .Case("jessie/sid", pylir::Distro::DebianJessie)
        .Case("stretch/sid", pylir::Distro::DebianStretch)
        .Case("buster/sid", pylir::Distro::DebianBuster)
        .Case("bullseye/sid", pylir::Distro::DebianBullseye)
        .Case("bookworm/sid", pylir::Distro::DebianBookworm)
        .Default(pylir::Distro::UnknownDistro);
  }

  // ...for SUSE
  file = llvm::MemoryBuffer::getFile("/etc/SuSE-release");
  if (file) {
    llvm::StringRef data = file.get()->getBuffer();
    llvm::SmallVector<llvm::StringRef, 8> lines;
    data.split(lines, "\n");
    for (const llvm::StringRef& line : lines) {
      if (!line.trim().starts_with("VERSION"))
        continue;

      std::pair<llvm::StringRef, llvm::StringRef> splitLine = line.split('=');
      // Old versions have split VERSION and PATCHLEVEL
      // Newer versions use VERSION = x.y
      std::pair<llvm::StringRef, llvm::StringRef> splitVer =
          splitLine.second.trim().split('.');
      int version;

      // OpenSUSE/SLES 10 and older are not supported and not compatible
      // with our rules, so just treat them as Distro::UnknownDistro.
      if (!splitVer.first.getAsInteger(10, version) && version > 10)
        return pylir::Distro::OpenSUSE;

      return pylir::Distro::UnknownDistro;
    }
    return pylir::Distro::UnknownDistro;
  }

  // ...and others.
  if (llvm::sys::fs::exists("/etc/gentoo-release"))
    return pylir::Distro::Gentoo;

  return pylir::Distro::UnknownDistro;
}

static pylir::Distro::DistroType getDistro(const llvm::Triple& TargetOrHost) {
  // If we don't target Linux, no need to check the distro. This saves a few
  // OS calls.
  if (!TargetOrHost.isOSLinux())
    return pylir::Distro::UnknownDistro;

  // If the host is not running Linux, and we're backed by a real file
  // system, no need to check the distro. This is the case where someone
  // is cross-compiling from BSD or Windows to Linux, and it would be
  // meaningless to try to figure out the "distro" of the non-Linux host.
  llvm::Triple hostTriple(llvm::sys::getProcessTriple());
  if (!hostTriple.isOSLinux())
    return pylir::Distro::UnknownDistro;

  static pylir::Distro::DistroType linuxDistro = detectDistro();
  return linuxDistro;
}

pylir::Distro::Distro(const llvm::Triple& TargetOrHost)
    : m_distroVal(getDistro(TargetOrHost)) {}
