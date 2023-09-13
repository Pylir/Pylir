//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- Distro.h - Linux distribution detection support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/TargetParser/Triple.h>

namespace pylir {

/// Distro - Helper class for detecting and classifying Linux distributions.
///
/// This class encapsulates the clang Linux distribution detection mechanism
/// as well as helper functions that match the specific (versioned) results
/// into wider distribution classes.
class Distro {
public:
  enum DistroType {
    // Special value means that no detection was performed yet.
    UninitializedDistro,
    // NB: Releases of a particular Linux distro should be kept together
    // in this enum, because some tests are done by integer comparison against
    // the first and last known member in the family, e.g. IsRedHat().
    AlpineLinux,
    ArchLinux,
    DebianLenny,
    DebianSqueeze,
    DebianWheezy,
    DebianJessie,
    DebianStretch,
    DebianBuster,
    DebianBullseye,
    DebianBookworm,
    Exherbo,
    RHEL5,
    RHEL6,
    RHEL7,
    Fedora,
    Gentoo,
    OpenSUSE,
    UbuntuHardy,
    UbuntuIntrepid,
    UbuntuJaunty,
    UbuntuKarmic,
    UbuntuLucid,
    UbuntuMaverick,
    UbuntuNatty,
    UbuntuOneiric,
    UbuntuPrecise,
    UbuntuQuantal,
    UbuntuRaring,
    UbuntuSaucy,
    UbuntuTrusty,
    UbuntuUtopic,
    UbuntuVivid,
    UbuntuWily,
    UbuntuXenial,
    UbuntuYakkety,
    UbuntuZesty,
    UbuntuArtful,
    UbuntuBionic,
    UbuntuCosmic,
    UbuntuDisco,
    UbuntuEoan,
    UbuntuFocal,
    UbuntuGroovy,
    UbuntuHirsute,
    UbuntuImpish,
    UbuntuJammy,
    UnknownDistro
  };

private:
  /// The distribution, possibly with specific version.
  DistroType m_distroVal;

public:
  /// @name Constructors
  /// @{

  /// Default constructor leaves the distribution unknown.
  Distro() : m_distroVal() {}

  /// Constructs a Distro type for specific distribution.
  explicit Distro(DistroType D) : m_distroVal(D) {}

  /// Detects the distribution using specified VFS.
  explicit Distro(const llvm::Triple& TargetOrHost);

  bool operator==(const Distro& Other) const {
    return m_distroVal == Other.m_distroVal;
  }

  bool operator==(const DistroType& Other) const {
    return m_distroVal == Other;
  }

  bool operator!=(const Distro& Other) const {
    return m_distroVal != Other.m_distroVal;
  }

  bool operator>=(const Distro& Other) const {
    return m_distroVal >= Other.m_distroVal;
  }

  bool operator>=(const DistroType& Other) const {
    return m_distroVal >= Other;
  }

  bool operator<=(const Distro& Other) const {
    return m_distroVal <= Other.m_distroVal;
  }

  [[nodiscard]] bool isRedhat() const {
    return m_distroVal == Fedora ||
           (m_distroVal >= RHEL5 && m_distroVal <= RHEL7);
  }

  [[nodiscard]] bool isOpenSuse() const {
    return m_distroVal == OpenSUSE;
  }

  [[nodiscard]] bool isDebian() const {
    return m_distroVal >= DebianLenny && m_distroVal <= DebianBookworm;
  }

  [[nodiscard]] bool isUbuntu() const {
    return m_distroVal >= UbuntuHardy && m_distroVal <= UbuntuJammy;
  }

  [[nodiscard]] bool isAlpineLinux() const {
    return m_distroVal == AlpineLinux;
  }

  [[nodiscard]] bool isGentoo() const {
    return m_distroVal == Gentoo;
  }
};

} // namespace pylir
