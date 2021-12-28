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
#include <llvm/ADT/Triple.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Threading.h>

static pylir::Distro::DistroType DetectOsRelease()
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File = llvm::MemoryBuffer::getFile("/etc/os-release");
    if (!File)
        File = llvm::MemoryBuffer::getFile("/usr/lib/os-release");
    if (!File)
        return pylir::Distro::UnknownDistro;

    llvm::SmallVector<llvm::StringRef, 16> Lines;
    File.get()->getBuffer().split(Lines, "\n");
    pylir::Distro::DistroType Version = pylir::Distro::UnknownDistro;

    // Obviously this can be improved a lot.
    for (llvm::StringRef Line : Lines)
        if (Version == pylir::Distro::UnknownDistro && Line.startswith("ID="))
            Version = llvm::StringSwitch<pylir::Distro::DistroType>(Line.substr(3))
                          .Case("alpine", pylir::Distro::AlpineLinux)
                          .Case("fedora", pylir::Distro::Fedora)
                          .Case("gentoo", pylir::Distro::Gentoo)
                          .Case("arch", pylir::Distro::ArchLinux)
                          // On SLES, /etc/os-release was introduced in SLES 11.
                          .Case("sles", pylir::Distro::OpenSUSE)
                          .Case("opensuse", pylir::Distro::OpenSUSE)
                          .Case("exherbo", pylir::Distro::Exherbo)
                          .Default(pylir::Distro::UnknownDistro);
    return Version;
}

static pylir::Distro::DistroType DetectLsbRelease()
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File = llvm::MemoryBuffer::getFile("/etc/lsb-release");
    if (!File)
        return pylir::Distro::UnknownDistro;

    llvm::SmallVector<llvm::StringRef, 16> Lines;
    File.get()->getBuffer().split(Lines, "\n");
    pylir::Distro::DistroType Version = pylir::Distro::UnknownDistro;

    for (llvm::StringRef Line : Lines)
        if (Version == pylir::Distro::UnknownDistro && Line.startswith("DISTRIB_CODENAME="))
            Version = llvm::StringSwitch<pylir::Distro::DistroType>(Line.substr(17))
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
    return Version;
}

static pylir::Distro::DistroType DetectDistro()
{
    pylir::Distro::DistroType Version = pylir::Distro::UnknownDistro;

    // Newer freedesktop.org's compilant systemd-based systems
    // should provide /etc/os-release or /usr/lib/os-release.
    Version = DetectOsRelease();
    if (Version != pylir::Distro::UnknownDistro)
        return Version;

    // Older systems might provide /etc/lsb-release.
    Version = DetectLsbRelease();
    if (Version != pylir::Distro::UnknownDistro)
        return Version;

    // Otherwise try some distro-specific quirks for RedHat...
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File = llvm::MemoryBuffer::getFile("/etc/redhat-release");

    if (File)
    {
        llvm::StringRef Data = File.get()->getBuffer();
        if (Data.startswith("Fedora release"))
            return pylir::Distro::Fedora;
        if (Data.startswith("Red Hat Enterprise Linux") || Data.startswith("CentOS")
            || Data.startswith("Scientific Linux"))
        {
            if (Data.contains("release 7"))
                return pylir::Distro::RHEL7;
            else if (Data.contains("release 6"))
                return pylir::Distro::RHEL6;
            else if (Data.contains("release 5"))
                return pylir::Distro::RHEL5;
        }
        return pylir::Distro::UnknownDistro;
    }

    // ...for Debian
    File = llvm::MemoryBuffer::getFile("/etc/debian_version");
    if (File)
    {
        llvm::StringRef Data = File.get()->getBuffer();
        // Contents: < major.minor > or < codename/sid >
        int MajorVersion;
        if (!Data.split('.').first.getAsInteger(10, MajorVersion))
        {
            switch (MajorVersion)
            {
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
        return llvm::StringSwitch<pylir::Distro::DistroType>(Data.split("\n").first)
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
    File = llvm::MemoryBuffer::getFile("/etc/SuSE-release");
    if (File)
    {
        llvm::StringRef Data = File.get()->getBuffer();
        llvm::SmallVector<llvm::StringRef, 8> Lines;
        Data.split(Lines, "\n");
        for (const llvm::StringRef& Line : Lines)
        {
            if (!Line.trim().startswith("VERSION"))
                continue;
            std::pair<llvm::StringRef, llvm::StringRef> SplitLine = Line.split('=');
            // Old versions have split VERSION and PATCHLEVEL
            // Newer versions use VERSION = x.y
            std::pair<llvm::StringRef, llvm::StringRef> SplitVer = SplitLine.second.trim().split('.');
            int Version;

            // OpenSUSE/SLES 10 and older are not supported and not compatible
            // with our rules, so just treat them as Distro::UnknownDistro.
            if (!SplitVer.first.getAsInteger(10, Version) && Version > 10)
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

static pylir::Distro::DistroType GetDistro(const llvm::Triple& TargetOrHost)
{
    // If we don't target Linux, no need to check the distro. This saves a few
    // OS calls.
    if (!TargetOrHost.isOSLinux())
        return pylir::Distro::UnknownDistro;

    // If the host is not running Linux, and we're backed by a real file
    // system, no need to check the distro. This is the case where someone
    // is cross-compiling from BSD or Windows to Linux, and it would be
    // meaningless to try to figure out the "distro" of the non-Linux host.
    llvm::Triple HostTriple(llvm::sys::getProcessTriple());
    if (!HostTriple.isOSLinux())
        return pylir::Distro::UnknownDistro;

    static pylir::Distro::DistroType LinuxDistro = DetectDistro();
    return LinuxDistro;
}

pylir::Distro::Distro(const llvm::Triple& TargetOrHost) : DistroVal(GetDistro(TargetOrHost)) {}
