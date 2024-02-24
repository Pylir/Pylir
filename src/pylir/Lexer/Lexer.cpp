//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Lexer.hpp"

#include <llvm/Support/UnicodeCharRanges.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Support/Util.hpp>

#include <charconv>
#include <functional>
#include <iterator>
#include <locale>
#include <unordered_map>

pylir::Lexer::Lexer(Diag::DiagnosticsDocManager<>& diagManager)
    : m_current(diagManager.getDocument().begin()),
      m_diagManager(&diagManager) {}

namespace {
#pragma region unicode
constexpr llvm::sys::UnicodeCharRange INITIAL_CHARACTERS[] = {
    {0x41, 0x5a},       {0x5f, 0x5f},       {0x61, 0x7a},
    {0xaa, 0xaa},       {0xb5, 0xb5},       {0xba, 0xba},
    {0xc0, 0xd6},       {0xd8, 0xf6},       {0xf8, 0x13e},
    {0x141, 0x2c1},     {0x2c6, 0x2d1},     {0x2e0, 0x2e4},
    {0x2ec, 0x2ec},     {0x2ee, 0x2ee},     {0x370, 0x374},
    {0x376, 0x377},     {0x37b, 0x37d},     {0x37f, 0x37f},
    {0x386, 0x386},     {0x388, 0x38a},     {0x38c, 0x38c},
    {0x38e, 0x3a1},     {0x3a3, 0x3f5},     {0x3f7, 0x481},
    {0x48a, 0x52f},     {0x531, 0x556},     {0x559, 0x559},
    {0x560, 0x588},     {0x5d0, 0x5ea},     {0x5ef, 0x5f2},
    {0x620, 0x64a},     {0x66e, 0x66f},     {0x671, 0x6d3},
    {0x6d5, 0x6d5},     {0x6e5, 0x6e6},     {0x6ee, 0x6ef},
    {0x6fa, 0x6fc},     {0x6ff, 0x6ff},     {0x710, 0x710},
    {0x712, 0x72f},     {0x74d, 0x7a5},     {0x7b1, 0x7b1},
    {0x7ca, 0x7ea},     {0x7f4, 0x7f5},     {0x7fa, 0x7fa},
    {0x800, 0x815},     {0x81a, 0x81a},     {0x824, 0x824},
    {0x828, 0x828},     {0x840, 0x858},     {0x860, 0x86a},
    {0x8a0, 0x8b4},     {0x8b6, 0x8bd},     {0x904, 0x939},
    {0x93d, 0x93d},     {0x950, 0x950},     {0x958, 0x961},
    {0x971, 0x980},     {0x985, 0x98c},     {0x98f, 0x990},
    {0x993, 0x9a8},     {0x9aa, 0x9b0},     {0x9b2, 0x9b2},
    {0x9b6, 0x9b9},     {0x9bd, 0x9bd},     {0x9ce, 0x9ce},
    {0x9dc, 0x9dd},     {0x9df, 0x9e1},     {0x9f0, 0x9f1},
    {0x9fc, 0x9fc},     {0xa05, 0xa0a},     {0xa0f, 0xa10},
    {0xa13, 0xa28},     {0xa2a, 0xa30},     {0xa32, 0xa33},
    {0xa35, 0xa36},     {0xa38, 0xa39},     {0xa59, 0xa5c},
    {0xa5e, 0xa5e},     {0xa72, 0xa74},     {0xa85, 0xa8d},
    {0xa8f, 0xa91},     {0xa93, 0xaa8},     {0xaaa, 0xab0},
    {0xab2, 0xab3},     {0xab5, 0xab9},     {0xabd, 0xabd},
    {0xad0, 0xad0},     {0xae0, 0xae1},     {0xaf9, 0xaf9},
    {0xb05, 0xb0c},     {0xb0f, 0xb10},     {0xb13, 0xb28},
    {0xb2a, 0xb30},     {0xb32, 0xb33},     {0xb35, 0xb39},
    {0xb3d, 0xb3d},     {0xb5c, 0xb5d},     {0xb5f, 0xb61},
    {0xb71, 0xb71},     {0xb83, 0xb83},     {0xb85, 0xb8a},
    {0xb8e, 0xb90},     {0xb92, 0xb95},     {0xb99, 0xb9a},
    {0xb9c, 0xb9c},     {0xb9e, 0xb9f},     {0xba3, 0xba4},
    {0xba8, 0xbaa},     {0xbae, 0xbb9},     {0xbd0, 0xbd0},
    {0xc05, 0xc0c},     {0xc0e, 0xc10},     {0xc12, 0xc28},
    {0xc2a, 0xc39},     {0xc3d, 0xc3d},     {0xc58, 0xc5a},
    {0xc60, 0xc61},     {0xc80, 0xc80},     {0xc85, 0xc8c},
    {0xc8e, 0xc90},     {0xc92, 0xca8},     {0xcaa, 0xcb3},
    {0xcb5, 0xcb9},     {0xcbd, 0xcbd},     {0xcde, 0xcde},
    {0xce0, 0xce1},     {0xcf1, 0xcf2},     {0xd05, 0xd0c},
    {0xd0e, 0xd10},     {0xd12, 0xd3a},     {0xd3d, 0xd3d},
    {0xd4e, 0xd4e},     {0xd54, 0xd56},     {0xd5f, 0xd61},
    {0xd7a, 0xd7f},     {0xd85, 0xd96},     {0xd9a, 0xdb1},
    {0xdb3, 0xdbb},     {0xdbd, 0xdbd},     {0xdc0, 0xdc6},
    {0xe01, 0xe30},     {0xe32, 0xe32},     {0xe40, 0xe46},
    {0xe81, 0xe82},     {0xe84, 0xe84},     {0xe86, 0xe8a},
    {0xe8c, 0xea3},     {0xea5, 0xea5},     {0xea7, 0xeb0},
    {0xeb2, 0xeb2},     {0xebd, 0xebd},     {0xec0, 0xec4},
    {0xec6, 0xec6},     {0xedc, 0xedf},     {0xf00, 0xf00},
    {0xf40, 0xf47},     {0xf49, 0xf6c},     {0xf88, 0xf8c},
    {0x1000, 0x102a},   {0x103f, 0x103f},   {0x1050, 0x1055},
    {0x105a, 0x105d},   {0x1061, 0x1061},   {0x1065, 0x1066},
    {0x106e, 0x1070},   {0x1075, 0x1081},   {0x108e, 0x108e},
    {0x10a0, 0x10c5},   {0x10c7, 0x10c7},   {0x10cd, 0x10cd},
    {0x10d0, 0x10fa},   {0x10fc, 0x1248},   {0x124a, 0x124d},
    {0x1250, 0x1256},   {0x1258, 0x1258},   {0x125a, 0x125d},
    {0x1260, 0x1288},   {0x128a, 0x128d},   {0x1290, 0x12b0},
    {0x12b2, 0x12b5},   {0x12b8, 0x12be},   {0x12c0, 0x12c0},
    {0x12c2, 0x12c5},   {0x12c8, 0x12d6},   {0x12d8, 0x1310},
    {0x1312, 0x1315},   {0x1318, 0x135a},   {0x1380, 0x138f},
    {0x13a0, 0x13f5},   {0x13f8, 0x13fd},   {0x1401, 0x166c},
    {0x166f, 0x167f},   {0x1681, 0x169a},   {0x16a0, 0x16ea},
    {0x16ee, 0x16f8},   {0x1700, 0x170c},   {0x170e, 0x1711},
    {0x1720, 0x1731},   {0x1740, 0x1751},   {0x1760, 0x176c},
    {0x176e, 0x1770},   {0x1780, 0x17b3},   {0x17d7, 0x17d7},
    {0x17dc, 0x17dc},   {0x1820, 0x1878},   {0x1880, 0x1884},
    {0x1887, 0x18a8},   {0x18aa, 0x18aa},   {0x18b0, 0x18f5},
    {0x1900, 0x191e},   {0x1950, 0x196d},   {0x1970, 0x1974},
    {0x1980, 0x19ab},   {0x19b0, 0x19c9},   {0x1a00, 0x1a16},
    {0x1a20, 0x1a54},   {0x1aa7, 0x1aa7},   {0x1b05, 0x1b33},
    {0x1b45, 0x1b4b},   {0x1b83, 0x1ba0},   {0x1bae, 0x1baf},
    {0x1bba, 0x1be5},   {0x1c00, 0x1c23},   {0x1c4d, 0x1c4f},
    {0x1c5a, 0x1c7d},   {0x1c80, 0x1c88},   {0x1c90, 0x1cba},
    {0x1cbd, 0x1cbf},   {0x1ce9, 0x1cec},   {0x1cee, 0x1cf3},
    {0x1cf5, 0x1cf6},   {0x1cfa, 0x1cfa},   {0x1d00, 0x1dbf},
    {0x1e00, 0x1f15},   {0x1f18, 0x1f1d},   {0x1f20, 0x1f45},
    {0x1f48, 0x1f4d},   {0x1f50, 0x1f57},   {0x1f59, 0x1f59},
    {0x1f5b, 0x1f5b},   {0x1f5d, 0x1f5d},   {0x1f5f, 0x1f7d},
    {0x1f80, 0x1fb4},   {0x1fb6, 0x1fbc},   {0x1fbe, 0x1fbe},
    {0x1fc2, 0x1fc4},   {0x1fc6, 0x1fcc},   {0x1fd0, 0x1fd3},
    {0x1fd6, 0x1fdb},   {0x1fe0, 0x1fec},   {0x1ff2, 0x1ff4},
    {0x1ff6, 0x1ffc},   {0x2071, 0x2071},   {0x207f, 0x207f},
    {0x2090, 0x209c},   {0x2102, 0x2102},   {0x2107, 0x2107},
    {0x210a, 0x2113},   {0x2115, 0x2115},   {0x2119, 0x211d},
    {0x2124, 0x2124},   {0x2126, 0x2126},   {0x2128, 0x2128},
    {0x212a, 0x212d},   {0x212f, 0x2139},   {0x213c, 0x213f},
    {0x2145, 0x2149},   {0x214e, 0x214e},   {0x2160, 0x2188},
    {0x2c00, 0x2c2e},   {0x2c30, 0x2c5e},   {0x2c60, 0x2ce4},
    {0x2ceb, 0x2cee},   {0x2cf2, 0x2cf3},   {0x2d00, 0x2d25},
    {0x2d27, 0x2d27},   {0x2d2d, 0x2d2d},   {0x2d30, 0x2d67},
    {0x2d6f, 0x2d6f},   {0x2d80, 0x2d96},   {0x2da0, 0x2da6},
    {0x2da8, 0x2dae},   {0x2db0, 0x2db6},   {0x2db8, 0x2dbe},
    {0x2dc0, 0x2dc6},   {0x2dc8, 0x2dce},   {0x2dd0, 0x2dd6},
    {0x2dd8, 0x2dde},   {0x2e2f, 0x2e2f},   {0x3005, 0x3007},
    {0x3021, 0x3029},   {0x3031, 0x3035},   {0x3038, 0x303c},
    {0x3041, 0x3096},   {0x309d, 0x309f},   {0x30a1, 0x30fa},
    {0x30fc, 0x30ff},   {0x3105, 0x312f},   {0x3131, 0x318e},
    {0x31a0, 0x31ba},   {0x31f0, 0x31ff},   {0x3400, 0x4db5},
    {0x4e00, 0x9fef},   {0xa000, 0xa48c},   {0xa4d0, 0xa4fd},
    {0xa500, 0xa60c},   {0xa610, 0xa61f},   {0xa62a, 0xa62b},
    {0xa640, 0xa66e},   {0xa67f, 0xa69d},   {0xa6a0, 0xa6ef},
    {0xa717, 0xa71f},   {0xa722, 0xa788},   {0xa78b, 0xa7bf},
    {0xa7c2, 0xa7c6},   {0xa7f7, 0xa801},   {0xa803, 0xa805},
    {0xa807, 0xa80a},   {0xa80c, 0xa822},   {0xa840, 0xa873},
    {0xa882, 0xa8b3},   {0xa8f2, 0xa8f7},   {0xa8fb, 0xa8fb},
    {0xa8fd, 0xa8fe},   {0xa90a, 0xa925},   {0xa930, 0xa946},
    {0xa960, 0xa97c},   {0xa984, 0xa9b2},   {0xa9cf, 0xa9cf},
    {0xa9e0, 0xa9e4},   {0xa9e6, 0xa9ef},   {0xa9fa, 0xa9fe},
    {0xaa00, 0xaa28},   {0xaa40, 0xaa42},   {0xaa44, 0xaa4b},
    {0xaa60, 0xaa76},   {0xaa7a, 0xaa7a},   {0xaa7e, 0xaaaf},
    {0xaab1, 0xaab1},   {0xaab5, 0xaab6},   {0xaab9, 0xaabd},
    {0xaac0, 0xaac0},   {0xaac2, 0xaac2},   {0xaadb, 0xaadd},
    {0xaae0, 0xaaea},   {0xaaf2, 0xaaf4},   {0xab01, 0xab06},
    {0xab09, 0xab0e},   {0xab11, 0xab16},   {0xab20, 0xab26},
    {0xab28, 0xab2e},   {0xab30, 0xab5a},   {0xab5c, 0xab67},
    {0xab70, 0xabe2},   {0xac00, 0xd7a3},   {0xd7b0, 0xd7c6},
    {0xd7cb, 0xd7fb},   {0xf900, 0xfa6d},   {0xfa70, 0xfad9},
    {0xfb00, 0xfb06},   {0xfb13, 0xfb17},   {0xfb1d, 0xfb1d},
    {0xfb1f, 0xfb28},   {0xfb2a, 0xfb36},   {0xfb38, 0xfb3c},
    {0xfb3e, 0xfb3e},   {0xfb40, 0xfb41},   {0xfb43, 0xfb44},
    {0xfb46, 0xfbb1},   {0xfbd3, 0xfc5d},   {0xfc64, 0xfd3d},
    {0xfd50, 0xfd8f},   {0xfd92, 0xfdc7},   {0xfdf0, 0xfdf9},
    {0xfe71, 0xfe71},   {0xfe73, 0xfe73},   {0xfe77, 0xfe77},
    {0xfe79, 0xfe79},   {0xfe7b, 0xfe7b},   {0xfe7d, 0xfe7d},
    {0xfe7f, 0xfefc},   {0xff21, 0xff3a},   {0xff41, 0xff5a},
    {0xff66, 0xff9d},   {0xffa0, 0xffbe},   {0xffc2, 0xffc7},
    {0xffca, 0xffcf},   {0xffd2, 0xffd7},   {0xffda, 0xffdc},
    {0x10000, 0x1000b}, {0x1000d, 0x10026}, {0x10028, 0x1003a},
    {0x1003c, 0x1003d}, {0x1003f, 0x1004d}, {0x10050, 0x1005d},
    {0x10080, 0x100fa}, {0x10140, 0x10174}, {0x10280, 0x1029c},
    {0x102a0, 0x102d0}, {0x10300, 0x1031f}, {0x1032d, 0x1034a},
    {0x10350, 0x10375}, {0x10380, 0x1039d}, {0x103a0, 0x103c3},
    {0x103c8, 0x103cf}, {0x103d1, 0x103d5}, {0x10400, 0x1049d},
    {0x104b0, 0x104d3}, {0x104d8, 0x104fb}, {0x10500, 0x10527},
    {0x10530, 0x10563}, {0x10600, 0x10736}, {0x10740, 0x10755},
    {0x10760, 0x10767}, {0x10800, 0x10805}, {0x10808, 0x10808},
    {0x1080a, 0x10835}, {0x10837, 0x10838}, {0x1083c, 0x1083c},
    {0x1083f, 0x10855}, {0x10860, 0x10876}, {0x10880, 0x1089e},
    {0x108e0, 0x108f2}, {0x108f4, 0x108f5}, {0x10900, 0x10915},
    {0x10920, 0x10939}, {0x10980, 0x109b7}, {0x109be, 0x109bf},
    {0x10a00, 0x10a00}, {0x10a10, 0x10a13}, {0x10a15, 0x10a17},
    {0x10a19, 0x10a35}, {0x10a60, 0x10a7c}, {0x10a80, 0x10a9c},
    {0x10ac0, 0x10ac7}, {0x10ac9, 0x10ae4}, {0x10b00, 0x10b35},
    {0x10b40, 0x10b55}, {0x10b60, 0x10b72}, {0x10b80, 0x10b91},
    {0x10c00, 0x10c48}, {0x10c80, 0x10cb2}, {0x10cc0, 0x10cf2},
    {0x10d00, 0x10d23}, {0x10f00, 0x10f1c}, {0x10f27, 0x10f27},
    {0x10f30, 0x10f45}, {0x10fe0, 0x10ff6}, {0x11003, 0x11037},
    {0x11083, 0x110af}, {0x110d0, 0x110e8}, {0x11103, 0x11126},
    {0x11144, 0x11144}, {0x11150, 0x11172}, {0x11176, 0x11176},
    {0x11183, 0x111b2}, {0x111c1, 0x111c4}, {0x111da, 0x111da},
    {0x111dc, 0x111dc}, {0x11200, 0x11211}, {0x11213, 0x1122b},
    {0x11280, 0x11286}, {0x11288, 0x11288}, {0x1128a, 0x1128d},
    {0x1128f, 0x1129d}, {0x1129f, 0x112a8}, {0x112b0, 0x112de},
    {0x11305, 0x1130c}, {0x1130f, 0x11310}, {0x11313, 0x11328},
    {0x1132a, 0x11330}, {0x11332, 0x11333}, {0x11335, 0x11339},
    {0x1133d, 0x1133d}, {0x11350, 0x11350}, {0x1135d, 0x11361},
    {0x11400, 0x11434}, {0x11447, 0x1144a}, {0x1145f, 0x1145f},
    {0x11480, 0x114af}, {0x114c4, 0x114c5}, {0x114c7, 0x114c7},
    {0x11580, 0x115ae}, {0x115d8, 0x115db}, {0x11600, 0x1162f},
    {0x11644, 0x11644}, {0x11680, 0x116aa}, {0x116b8, 0x116b8},
    {0x11700, 0x1171a}, {0x11800, 0x1182b}, {0x118a0, 0x118df},
    {0x118ff, 0x118ff}, {0x119a0, 0x119a7}, {0x119aa, 0x119d0},
    {0x119e1, 0x119e1}, {0x119e3, 0x119e3}, {0x11a00, 0x11a00},
    {0x11a0b, 0x11a32}, {0x11a3a, 0x11a3a}, {0x11a50, 0x11a50},
    {0x11a5c, 0x11a89}, {0x11a9d, 0x11a9d}, {0x11ac0, 0x11af8},
    {0x11c00, 0x11c08}, {0x11c0a, 0x11c2e}, {0x11c40, 0x11c40},
    {0x11c72, 0x11c8f}, {0x11d00, 0x11d06}, {0x11d08, 0x11d09},
    {0x11d0b, 0x11d30}, {0x11d46, 0x11d46}, {0x11d60, 0x11d65},
    {0x11d67, 0x11d68}, {0x11d6a, 0x11d89}, {0x11d98, 0x11d98},
    {0x11ee0, 0x11ef2}, {0x12000, 0x12399}, {0x12400, 0x1246e},
    {0x12480, 0x12543}, {0x13000, 0x1342e}, {0x14400, 0x14646},
    {0x16800, 0x16a38}, {0x16a40, 0x16a5e}, {0x16ad0, 0x16aed},
    {0x16b00, 0x16b2f}, {0x16b40, 0x16b43}, {0x16b63, 0x16b77},
    {0x16b7d, 0x16b8f}, {0x16e40, 0x16e7f}, {0x16f00, 0x16f4a},
    {0x16f50, 0x16f50}, {0x16f93, 0x16f9f}, {0x16fe0, 0x16fe1},
    {0x16fe3, 0x16fe3}, {0x17000, 0x187f7}, {0x18800, 0x18af2},
    {0x1b000, 0x1b11e}, {0x1b150, 0x1b152}, {0x1b164, 0x1b167},
    {0x1b170, 0x1b2fb}, {0x1bc00, 0x1bc6a}, {0x1bc70, 0x1bc7c},
    {0x1bc80, 0x1bc88}, {0x1bc90, 0x1bc99}, {0x1d400, 0x1d454},
    {0x1d456, 0x1d49c}, {0x1d49e, 0x1d49f}, {0x1d4a2, 0x1d4a2},
    {0x1d4a5, 0x1d4a6}, {0x1d4a9, 0x1d4ac}, {0x1d4ae, 0x1d4b9},
    {0x1d4bb, 0x1d4bb}, {0x1d4bd, 0x1d4c3}, {0x1d4c5, 0x1d505},
    {0x1d507, 0x1d50a}, {0x1d50d, 0x1d514}, {0x1d516, 0x1d51c},
    {0x1d51e, 0x1d539}, {0x1d53b, 0x1d53e}, {0x1d540, 0x1d544},
    {0x1d546, 0x1d546}, {0x1d54a, 0x1d550}, {0x1d552, 0x1d6a5},
    {0x1d6a8, 0x1d6c0}, {0x1d6c2, 0x1d6da}, {0x1d6dc, 0x1d6fa},
    {0x1d6fc, 0x1d714}, {0x1d716, 0x1d734}, {0x1d736, 0x1d74e},
    {0x1d750, 0x1d76e}, {0x1d770, 0x1d788}, {0x1d78a, 0x1d7a8},
    {0x1d7aa, 0x1d7c2}, {0x1d7c4, 0x1d7cb}, {0x1e100, 0x1e12c},
    {0x1e137, 0x1e13d}, {0x1e14e, 0x1e14e}, {0x1e2c0, 0x1e2eb},
    {0x1e800, 0x1e8c4}, {0x1e900, 0x1e943}, {0x1e94b, 0x1e94b},
    {0x1ee00, 0x1ee03}, {0x1ee05, 0x1ee1f}, {0x1ee21, 0x1ee22},
    {0x1ee24, 0x1ee24}, {0x1ee27, 0x1ee27}, {0x1ee29, 0x1ee32},
    {0x1ee34, 0x1ee37}, {0x1ee39, 0x1ee39}, {0x1ee3b, 0x1ee3b},
    {0x1ee42, 0x1ee42}, {0x1ee47, 0x1ee47}, {0x1ee49, 0x1ee49},
    {0x1ee4b, 0x1ee4b}, {0x1ee4d, 0x1ee4f}, {0x1ee51, 0x1ee52},
    {0x1ee54, 0x1ee54}, {0x1ee57, 0x1ee57}, {0x1ee59, 0x1ee59},
    {0x1ee5b, 0x1ee5b}, {0x1ee5d, 0x1ee5d}, {0x1ee5f, 0x1ee5f},
    {0x1ee61, 0x1ee62}, {0x1ee64, 0x1ee64}, {0x1ee67, 0x1ee6a},
    {0x1ee6c, 0x1ee72}, {0x1ee74, 0x1ee77}, {0x1ee79, 0x1ee7c},
    {0x1ee7e, 0x1ee7e}, {0x1ee80, 0x1ee89}, {0x1ee8b, 0x1ee9b},
    {0x1eea1, 0x1eea3}, {0x1eea5, 0x1eea9}, {0x1eeab, 0x1eebb},
    {0x20000, 0x2a6d6}, {0x2a700, 0x2b734}, {0x2b740, 0x2b81d},
    {0x2b820, 0x2cea1}, {0x2ceb0, 0x2ebe0},
};
constexpr llvm::sys::UnicodeCharRange LEGAL_IDENTIFIERS[] = {
    {0x30, 0x39},       {0x41, 0x5a},       {0x5f, 0x5f},
    {0x61, 0x7a},       {0xaa, 0xaa},       {0xb5, 0xb5},
    {0xba, 0xba},       {0xc0, 0xd6},       {0xd8, 0xf6},
    {0xf8, 0x13e},      {0x141, 0x2c1},     {0x2c6, 0x2d1},
    {0x2e0, 0x2e4},     {0x2ec, 0x2ec},     {0x2ee, 0x2ee},
    {0x300, 0x374},     {0x376, 0x377},     {0x37b, 0x37d},
    {0x37f, 0x37f},     {0x386, 0x386},     {0x388, 0x38a},
    {0x38c, 0x38c},     {0x38e, 0x3a1},     {0x3a3, 0x3f5},
    {0x3f7, 0x481},     {0x483, 0x487},     {0x48a, 0x52f},
    {0x531, 0x556},     {0x559, 0x559},     {0x560, 0x588},
    {0x591, 0x5bd},     {0x5bf, 0x5bf},     {0x5c1, 0x5c2},
    {0x5c4, 0x5c5},     {0x5c7, 0x5c7},     {0x5d0, 0x5ea},
    {0x5ef, 0x5f2},     {0x610, 0x61a},     {0x620, 0x669},
    {0x66e, 0x6d3},     {0x6d5, 0x6dc},     {0x6df, 0x6e8},
    {0x6ea, 0x6fc},     {0x6ff, 0x6ff},     {0x710, 0x74a},
    {0x74d, 0x7b1},     {0x7c0, 0x7f5},     {0x7fa, 0x7fa},
    {0x7fd, 0x7fd},     {0x800, 0x82d},     {0x840, 0x85b},
    {0x860, 0x86a},     {0x8a0, 0x8b4},     {0x8b6, 0x8bd},
    {0x8d3, 0x8e1},     {0x8e3, 0x963},     {0x966, 0x96f},
    {0x971, 0x983},     {0x985, 0x98c},     {0x98f, 0x990},
    {0x993, 0x9a8},     {0x9aa, 0x9b0},     {0x9b2, 0x9b2},
    {0x9b6, 0x9b9},     {0x9bc, 0x9c4},     {0x9c7, 0x9c8},
    {0x9cb, 0x9ce},     {0x9d7, 0x9d7},     {0x9dc, 0x9dd},
    {0x9df, 0x9e3},     {0x9e6, 0x9f1},     {0x9fc, 0x9fc},
    {0x9fe, 0x9fe},     {0xa01, 0xa03},     {0xa05, 0xa0a},
    {0xa0f, 0xa10},     {0xa13, 0xa28},     {0xa2a, 0xa30},
    {0xa32, 0xa33},     {0xa35, 0xa36},     {0xa38, 0xa39},
    {0xa3c, 0xa3c},     {0xa3e, 0xa42},     {0xa47, 0xa48},
    {0xa4b, 0xa4d},     {0xa51, 0xa51},     {0xa59, 0xa5c},
    {0xa5e, 0xa5e},     {0xa66, 0xa75},     {0xa81, 0xa83},
    {0xa85, 0xa8d},     {0xa8f, 0xa91},     {0xa93, 0xaa8},
    {0xaaa, 0xab0},     {0xab2, 0xab3},     {0xab5, 0xab9},
    {0xabc, 0xac5},     {0xac7, 0xac9},     {0xacb, 0xacd},
    {0xad0, 0xad0},     {0xae0, 0xae3},     {0xae6, 0xaef},
    {0xaf9, 0xaff},     {0xb01, 0xb03},     {0xb05, 0xb0c},
    {0xb0f, 0xb10},     {0xb13, 0xb28},     {0xb2a, 0xb30},
    {0xb32, 0xb33},     {0xb35, 0xb39},     {0xb3c, 0xb44},
    {0xb47, 0xb48},     {0xb4b, 0xb4d},     {0xb56, 0xb57},
    {0xb5c, 0xb5d},     {0xb5f, 0xb63},     {0xb66, 0xb6f},
    {0xb71, 0xb71},     {0xb82, 0xb83},     {0xb85, 0xb8a},
    {0xb8e, 0xb90},     {0xb92, 0xb95},     {0xb99, 0xb9a},
    {0xb9c, 0xb9c},     {0xb9e, 0xb9f},     {0xba3, 0xba4},
    {0xba8, 0xbaa},     {0xbae, 0xbb9},     {0xbbe, 0xbc2},
    {0xbc6, 0xbc8},     {0xbca, 0xbcd},     {0xbd0, 0xbd0},
    {0xbd7, 0xbd7},     {0xbe6, 0xbef},     {0xc00, 0xc0c},
    {0xc0e, 0xc10},     {0xc12, 0xc28},     {0xc2a, 0xc39},
    {0xc3d, 0xc44},     {0xc46, 0xc48},     {0xc4a, 0xc4d},
    {0xc55, 0xc56},     {0xc58, 0xc5a},     {0xc60, 0xc63},
    {0xc66, 0xc6f},     {0xc80, 0xc83},     {0xc85, 0xc8c},
    {0xc8e, 0xc90},     {0xc92, 0xca8},     {0xcaa, 0xcb3},
    {0xcb5, 0xcb9},     {0xcbc, 0xcc4},     {0xcc6, 0xcc8},
    {0xcca, 0xccd},     {0xcd5, 0xcd6},     {0xcde, 0xcde},
    {0xce0, 0xce3},     {0xce6, 0xcef},     {0xcf1, 0xcf2},
    {0xd00, 0xd03},     {0xd05, 0xd0c},     {0xd0e, 0xd10},
    {0xd12, 0xd44},     {0xd46, 0xd48},     {0xd4a, 0xd4e},
    {0xd54, 0xd57},     {0xd5f, 0xd63},     {0xd66, 0xd6f},
    {0xd7a, 0xd7f},     {0xd82, 0xd83},     {0xd85, 0xd96},
    {0xd9a, 0xdb1},     {0xdb3, 0xdbb},     {0xdbd, 0xdbd},
    {0xdc0, 0xdc6},     {0xdca, 0xdca},     {0xdcf, 0xdd4},
    {0xdd6, 0xdd6},     {0xdd8, 0xddf},     {0xde6, 0xdef},
    {0xdf2, 0xdf3},     {0xe01, 0xe3a},     {0xe40, 0xe4e},
    {0xe50, 0xe59},     {0xe81, 0xe82},     {0xe84, 0xe84},
    {0xe86, 0xe8a},     {0xe8c, 0xea3},     {0xea5, 0xea5},
    {0xea7, 0xebd},     {0xec0, 0xec4},     {0xec6, 0xec6},
    {0xec8, 0xecd},     {0xed0, 0xed9},     {0xedc, 0xedf},
    {0xf00, 0xf00},     {0xf18, 0xf19},     {0xf20, 0xf29},
    {0xf35, 0xf35},     {0xf37, 0xf37},     {0xf39, 0xf39},
    {0xf3e, 0xf47},     {0xf49, 0xf6c},     {0xf71, 0xf84},
    {0xf86, 0xf97},     {0xf99, 0xfbc},     {0xfc6, 0xfc6},
    {0x1000, 0x1049},   {0x1050, 0x109d},   {0x10a0, 0x10c5},
    {0x10c7, 0x10c7},   {0x10cd, 0x10cd},   {0x10d0, 0x10fa},
    {0x10fc, 0x1248},   {0x124a, 0x124d},   {0x1250, 0x1256},
    {0x1258, 0x1258},   {0x125a, 0x125d},   {0x1260, 0x1288},
    {0x128a, 0x128d},   {0x1290, 0x12b0},   {0x12b2, 0x12b5},
    {0x12b8, 0x12be},   {0x12c0, 0x12c0},   {0x12c2, 0x12c5},
    {0x12c8, 0x12d6},   {0x12d8, 0x1310},   {0x1312, 0x1315},
    {0x1318, 0x135a},   {0x135d, 0x135f},   {0x1380, 0x138f},
    {0x13a0, 0x13f5},   {0x13f8, 0x13fd},   {0x1401, 0x166c},
    {0x166f, 0x167f},   {0x1681, 0x169a},   {0x16a0, 0x16ea},
    {0x16ee, 0x16f8},   {0x1700, 0x170c},   {0x170e, 0x1714},
    {0x1720, 0x1734},   {0x1740, 0x1753},   {0x1760, 0x176c},
    {0x176e, 0x1770},   {0x1772, 0x1773},   {0x1780, 0x17d3},
    {0x17d7, 0x17d7},   {0x17dc, 0x17dd},   {0x17e0, 0x17e9},
    {0x180b, 0x180d},   {0x1810, 0x1819},   {0x1820, 0x1878},
    {0x1880, 0x18aa},   {0x18b0, 0x18f5},   {0x1900, 0x191e},
    {0x1920, 0x192b},   {0x1930, 0x193b},   {0x1946, 0x196d},
    {0x1970, 0x1974},   {0x1980, 0x19ab},   {0x19b0, 0x19c9},
    {0x19d0, 0x19d9},   {0x1a00, 0x1a1b},   {0x1a20, 0x1a5e},
    {0x1a60, 0x1a7c},   {0x1a7f, 0x1a89},   {0x1a90, 0x1a99},
    {0x1aa7, 0x1aa7},   {0x1ab0, 0x1abd},   {0x1b00, 0x1b4b},
    {0x1b50, 0x1b59},   {0x1b6b, 0x1b73},   {0x1b80, 0x1bf3},
    {0x1c00, 0x1c37},   {0x1c40, 0x1c49},   {0x1c4d, 0x1c7d},
    {0x1c80, 0x1c88},   {0x1c90, 0x1cba},   {0x1cbd, 0x1cbf},
    {0x1cd0, 0x1cd2},   {0x1cd4, 0x1cfa},   {0x1d00, 0x1df9},
    {0x1dfb, 0x1f15},   {0x1f18, 0x1f1d},   {0x1f20, 0x1f45},
    {0x1f48, 0x1f4d},   {0x1f50, 0x1f57},   {0x1f59, 0x1f59},
    {0x1f5b, 0x1f5b},   {0x1f5d, 0x1f5d},   {0x1f5f, 0x1f7d},
    {0x1f80, 0x1fb4},   {0x1fb6, 0x1fbc},   {0x1fbe, 0x1fbe},
    {0x1fc2, 0x1fc4},   {0x1fc6, 0x1fcc},   {0x1fd0, 0x1fd3},
    {0x1fd6, 0x1fdb},   {0x1fe0, 0x1fec},   {0x1ff2, 0x1ff4},
    {0x1ff6, 0x1ffc},   {0x203f, 0x2040},   {0x2054, 0x2054},
    {0x2071, 0x2071},   {0x207f, 0x207f},   {0x2090, 0x209c},
    {0x20d0, 0x20dc},   {0x20e1, 0x20e1},   {0x20e5, 0x20f0},
    {0x2102, 0x2102},   {0x2107, 0x2107},   {0x210a, 0x2113},
    {0x2115, 0x2115},   {0x2119, 0x211d},   {0x2124, 0x2124},
    {0x2126, 0x2126},   {0x2128, 0x2128},   {0x212a, 0x212d},
    {0x212f, 0x2139},   {0x213c, 0x213f},   {0x2145, 0x2149},
    {0x214e, 0x214e},   {0x2160, 0x2188},   {0x2c00, 0x2c2e},
    {0x2c30, 0x2c5e},   {0x2c60, 0x2ce4},   {0x2ceb, 0x2cf3},
    {0x2d00, 0x2d25},   {0x2d27, 0x2d27},   {0x2d2d, 0x2d2d},
    {0x2d30, 0x2d67},   {0x2d6f, 0x2d6f},   {0x2d7f, 0x2d96},
    {0x2da0, 0x2da6},   {0x2da8, 0x2dae},   {0x2db0, 0x2db6},
    {0x2db8, 0x2dbe},   {0x2dc0, 0x2dc6},   {0x2dc8, 0x2dce},
    {0x2dd0, 0x2dd6},   {0x2dd8, 0x2dde},   {0x2de0, 0x2dff},
    {0x2e2f, 0x2e2f},   {0x3005, 0x3007},   {0x3021, 0x302f},
    {0x3031, 0x3035},   {0x3038, 0x303c},   {0x3041, 0x3096},
    {0x3099, 0x309a},   {0x309d, 0x309f},   {0x30a1, 0x30fa},
    {0x30fc, 0x30ff},   {0x3105, 0x312f},   {0x3131, 0x318e},
    {0x31a0, 0x31ba},   {0x31f0, 0x31ff},   {0x3400, 0x4db5},
    {0x4e00, 0x9fef},   {0xa000, 0xa48c},   {0xa4d0, 0xa4fd},
    {0xa500, 0xa60c},   {0xa610, 0xa62b},   {0xa640, 0xa66f},
    {0xa674, 0xa67d},   {0xa67f, 0xa6f1},   {0xa717, 0xa71f},
    {0xa722, 0xa788},   {0xa78b, 0xa7bf},   {0xa7c2, 0xa7c6},
    {0xa7f7, 0xa827},   {0xa840, 0xa873},   {0xa880, 0xa8c5},
    {0xa8d0, 0xa8d9},   {0xa8e0, 0xa8f7},   {0xa8fb, 0xa8fb},
    {0xa8fd, 0xa92d},   {0xa930, 0xa953},   {0xa960, 0xa97c},
    {0xa980, 0xa9c0},   {0xa9cf, 0xa9d9},   {0xa9e0, 0xa9fe},
    {0xaa00, 0xaa36},   {0xaa40, 0xaa4d},   {0xaa50, 0xaa59},
    {0xaa60, 0xaa76},   {0xaa7a, 0xaac2},   {0xaadb, 0xaadd},
    {0xaae0, 0xaaef},   {0xaaf2, 0xaaf6},   {0xab01, 0xab06},
    {0xab09, 0xab0e},   {0xab11, 0xab16},   {0xab20, 0xab26},
    {0xab28, 0xab2e},   {0xab30, 0xab5a},   {0xab5c, 0xab67},
    {0xab70, 0xabea},   {0xabec, 0xabed},   {0xabf0, 0xabf9},
    {0xac00, 0xd7a3},   {0xd7b0, 0xd7c6},   {0xd7cb, 0xd7fb},
    {0xf900, 0xfa6d},   {0xfa70, 0xfad9},   {0xfb00, 0xfb06},
    {0xfb13, 0xfb17},   {0xfb1d, 0xfb28},   {0xfb2a, 0xfb36},
    {0xfb38, 0xfb3c},   {0xfb3e, 0xfb3e},   {0xfb40, 0xfb41},
    {0xfb43, 0xfb44},   {0xfb46, 0xfbb1},   {0xfbd3, 0xfc5d},
    {0xfc64, 0xfd3d},   {0xfd50, 0xfd8f},   {0xfd92, 0xfdc7},
    {0xfdf0, 0xfdf9},   {0xfe00, 0xfe0f},   {0xfe20, 0xfe2f},
    {0xfe33, 0xfe34},   {0xfe4d, 0xfe4f},   {0xfe71, 0xfe71},
    {0xfe73, 0xfe73},   {0xfe77, 0xfe77},   {0xfe79, 0xfe79},
    {0xfe7b, 0xfe7b},   {0xfe7d, 0xfe7d},   {0xfe7f, 0xfefc},
    {0xff10, 0xff19},   {0xff21, 0xff3a},   {0xff3f, 0xff3f},
    {0xff41, 0xff5a},   {0xff66, 0xffbe},   {0xffc2, 0xffc7},
    {0xffca, 0xffcf},   {0xffd2, 0xffd7},   {0xffda, 0xffdc},
    {0x10000, 0x1000b}, {0x1000d, 0x10026}, {0x10028, 0x1003a},
    {0x1003c, 0x1003d}, {0x1003f, 0x1004d}, {0x10050, 0x1005d},
    {0x10080, 0x100fa}, {0x10140, 0x10174}, {0x101fd, 0x101fd},
    {0x10280, 0x1029c}, {0x102a0, 0x102d0}, {0x102e0, 0x102e0},
    {0x10300, 0x1031f}, {0x1032d, 0x1034a}, {0x10350, 0x1037a},
    {0x10380, 0x1039d}, {0x103a0, 0x103c3}, {0x103c8, 0x103cf},
    {0x103d1, 0x103d5}, {0x10400, 0x1049d}, {0x104a0, 0x104a9},
    {0x104b0, 0x104d3}, {0x104d8, 0x104fb}, {0x10500, 0x10527},
    {0x10530, 0x10563}, {0x10600, 0x10736}, {0x10740, 0x10755},
    {0x10760, 0x10767}, {0x10800, 0x10805}, {0x10808, 0x10808},
    {0x1080a, 0x10835}, {0x10837, 0x10838}, {0x1083c, 0x1083c},
    {0x1083f, 0x10855}, {0x10860, 0x10876}, {0x10880, 0x1089e},
    {0x108e0, 0x108f2}, {0x108f4, 0x108f5}, {0x10900, 0x10915},
    {0x10920, 0x10939}, {0x10980, 0x109b7}, {0x109be, 0x109bf},
    {0x10a00, 0x10a03}, {0x10a05, 0x10a06}, {0x10a0c, 0x10a13},
    {0x10a15, 0x10a17}, {0x10a19, 0x10a35}, {0x10a38, 0x10a3a},
    {0x10a3f, 0x10a3f}, {0x10a60, 0x10a7c}, {0x10a80, 0x10a9c},
    {0x10ac0, 0x10ac7}, {0x10ac9, 0x10ae6}, {0x10b00, 0x10b35},
    {0x10b40, 0x10b55}, {0x10b60, 0x10b72}, {0x10b80, 0x10b91},
    {0x10c00, 0x10c48}, {0x10c80, 0x10cb2}, {0x10cc0, 0x10cf2},
    {0x10d00, 0x10d27}, {0x10d30, 0x10d39}, {0x10f00, 0x10f1c},
    {0x10f27, 0x10f27}, {0x10f30, 0x10f50}, {0x10fe0, 0x10ff6},
    {0x11000, 0x11046}, {0x11066, 0x1106f}, {0x1107f, 0x110ba},
    {0x110d0, 0x110e8}, {0x110f0, 0x110f9}, {0x11100, 0x11134},
    {0x11136, 0x1113f}, {0x11144, 0x11146}, {0x11150, 0x11173},
    {0x11176, 0x11176}, {0x11180, 0x111c4}, {0x111c9, 0x111cc},
    {0x111d0, 0x111da}, {0x111dc, 0x111dc}, {0x11200, 0x11211},
    {0x11213, 0x11237}, {0x1123e, 0x1123e}, {0x11280, 0x11286},
    {0x11288, 0x11288}, {0x1128a, 0x1128d}, {0x1128f, 0x1129d},
    {0x1129f, 0x112a8}, {0x112b0, 0x112ea}, {0x112f0, 0x112f9},
    {0x11300, 0x11303}, {0x11305, 0x1130c}, {0x1130f, 0x11310},
    {0x11313, 0x11328}, {0x1132a, 0x11330}, {0x11332, 0x11333},
    {0x11335, 0x11339}, {0x1133b, 0x11344}, {0x11347, 0x11348},
    {0x1134b, 0x1134d}, {0x11350, 0x11350}, {0x11357, 0x11357},
    {0x1135d, 0x11363}, {0x11366, 0x1136c}, {0x11370, 0x11374},
    {0x11400, 0x1144a}, {0x11450, 0x11459}, {0x1145e, 0x1145f},
    {0x11480, 0x114c5}, {0x114c7, 0x114c7}, {0x114d0, 0x114d9},
    {0x11580, 0x115b5}, {0x115b8, 0x115c0}, {0x115d8, 0x115dd},
    {0x11600, 0x11640}, {0x11644, 0x11644}, {0x11650, 0x11659},
    {0x11680, 0x116b8}, {0x116c0, 0x116c9}, {0x11700, 0x1171a},
    {0x1171d, 0x1172b}, {0x11730, 0x11739}, {0x11800, 0x1183a},
    {0x118a0, 0x118e9}, {0x118ff, 0x118ff}, {0x119a0, 0x119a7},
    {0x119aa, 0x119d7}, {0x119da, 0x119e1}, {0x119e3, 0x119e4},
    {0x11a00, 0x11a3e}, {0x11a47, 0x11a47}, {0x11a50, 0x11a99},
    {0x11a9d, 0x11a9d}, {0x11ac0, 0x11af8}, {0x11c00, 0x11c08},
    {0x11c0a, 0x11c36}, {0x11c38, 0x11c40}, {0x11c50, 0x11c59},
    {0x11c72, 0x11c8f}, {0x11c92, 0x11ca7}, {0x11ca9, 0x11cb6},
    {0x11d00, 0x11d06}, {0x11d08, 0x11d09}, {0x11d0b, 0x11d36},
    {0x11d3a, 0x11d3a}, {0x11d3c, 0x11d3d}, {0x11d3f, 0x11d47},
    {0x11d50, 0x11d59}, {0x11d60, 0x11d65}, {0x11d67, 0x11d68},
    {0x11d6a, 0x11d8e}, {0x11d90, 0x11d91}, {0x11d93, 0x11d98},
    {0x11da0, 0x11da9}, {0x11ee0, 0x11ef6}, {0x12000, 0x12399},
    {0x12400, 0x1246e}, {0x12480, 0x12543}, {0x13000, 0x1342e},
    {0x14400, 0x14646}, {0x16800, 0x16a38}, {0x16a40, 0x16a5e},
    {0x16a60, 0x16a69}, {0x16ad0, 0x16aed}, {0x16af0, 0x16af4},
    {0x16b00, 0x16b36}, {0x16b40, 0x16b43}, {0x16b50, 0x16b59},
    {0x16b63, 0x16b77}, {0x16b7d, 0x16b8f}, {0x16e40, 0x16e7f},
    {0x16f00, 0x16f4a}, {0x16f4f, 0x16f87}, {0x16f8f, 0x16f9f},
    {0x16fe0, 0x16fe1}, {0x16fe3, 0x16fe3}, {0x17000, 0x187f7},
    {0x18800, 0x18af2}, {0x1b000, 0x1b11e}, {0x1b150, 0x1b152},
    {0x1b164, 0x1b167}, {0x1b170, 0x1b2fb}, {0x1bc00, 0x1bc6a},
    {0x1bc70, 0x1bc7c}, {0x1bc80, 0x1bc88}, {0x1bc90, 0x1bc99},
    {0x1bc9d, 0x1bc9e}, {0x1d165, 0x1d169}, {0x1d16d, 0x1d172},
    {0x1d17b, 0x1d182}, {0x1d185, 0x1d18b}, {0x1d1aa, 0x1d1ad},
    {0x1d242, 0x1d244}, {0x1d400, 0x1d454}, {0x1d456, 0x1d49c},
    {0x1d49e, 0x1d49f}, {0x1d4a2, 0x1d4a2}, {0x1d4a5, 0x1d4a6},
    {0x1d4a9, 0x1d4ac}, {0x1d4ae, 0x1d4b9}, {0x1d4bb, 0x1d4bb},
    {0x1d4bd, 0x1d4c3}, {0x1d4c5, 0x1d505}, {0x1d507, 0x1d50a},
    {0x1d50d, 0x1d514}, {0x1d516, 0x1d51c}, {0x1d51e, 0x1d539},
    {0x1d53b, 0x1d53e}, {0x1d540, 0x1d544}, {0x1d546, 0x1d546},
    {0x1d54a, 0x1d550}, {0x1d552, 0x1d6a5}, {0x1d6a8, 0x1d6c0},
    {0x1d6c2, 0x1d6da}, {0x1d6dc, 0x1d6fa}, {0x1d6fc, 0x1d714},
    {0x1d716, 0x1d734}, {0x1d736, 0x1d74e}, {0x1d750, 0x1d76e},
    {0x1d770, 0x1d788}, {0x1d78a, 0x1d7a8}, {0x1d7aa, 0x1d7c2},
    {0x1d7c4, 0x1d7cb}, {0x1d7ce, 0x1d7ff}, {0x1da00, 0x1da36},
    {0x1da3b, 0x1da6c}, {0x1da75, 0x1da75}, {0x1da84, 0x1da84},
    {0x1da9b, 0x1da9f}, {0x1daa1, 0x1daaf}, {0x1e000, 0x1e006},
    {0x1e008, 0x1e018}, {0x1e01b, 0x1e021}, {0x1e023, 0x1e024},
    {0x1e026, 0x1e02a}, {0x1e100, 0x1e12c}, {0x1e130, 0x1e13d},
    {0x1e140, 0x1e149}, {0x1e14e, 0x1e14e}, {0x1e2c0, 0x1e2f9},
    {0x1e800, 0x1e8c4}, {0x1e8d0, 0x1e8d6}, {0x1e900, 0x1e94b},
    {0x1e950, 0x1e959}, {0x1ee00, 0x1ee03}, {0x1ee05, 0x1ee1f},
    {0x1ee21, 0x1ee22}, {0x1ee24, 0x1ee24}, {0x1ee27, 0x1ee27},
    {0x1ee29, 0x1ee32}, {0x1ee34, 0x1ee37}, {0x1ee39, 0x1ee39},
    {0x1ee3b, 0x1ee3b}, {0x1ee42, 0x1ee42}, {0x1ee47, 0x1ee47},
    {0x1ee49, 0x1ee49}, {0x1ee4b, 0x1ee4b}, {0x1ee4d, 0x1ee4f},
    {0x1ee51, 0x1ee52}, {0x1ee54, 0x1ee54}, {0x1ee57, 0x1ee57},
    {0x1ee59, 0x1ee59}, {0x1ee5b, 0x1ee5b}, {0x1ee5d, 0x1ee5d},
    {0x1ee5f, 0x1ee5f}, {0x1ee61, 0x1ee62}, {0x1ee64, 0x1ee64},
    {0x1ee67, 0x1ee6a}, {0x1ee6c, 0x1ee72}, {0x1ee74, 0x1ee77},
    {0x1ee79, 0x1ee7c}, {0x1ee7e, 0x1ee7e}, {0x1ee80, 0x1ee89},
    {0x1ee8b, 0x1ee9b}, {0x1eea1, 0x1eea3}, {0x1eea5, 0x1eea9},
    {0x1eeab, 0x1eebb}, {0x20000, 0x2a6d6}, {0x2a700, 0x2b734},
    {0x2b740, 0x2b81d}, {0x2b820, 0x2cea1}, {0x2ceb0, 0x2ebe0},
    {0x2f800, 0x2fa1d},
};
#pragma endregion

bool isBlankLineSoFar(llvm::ArrayRef<pylir::Token> tokens) {
  return tokens.empty() ||
         tokens.back().getTokenType() == pylir::TokenType::Newline;
}

} // namespace

bool pylir::Lexer::parseNext() {
  if (m_current == m_diagManager->getDocument().end())
    return false;

  std::size_t startSize = m_tokens.size();
  do {
    const auto* start = m_current;
    switch (*m_current) {
    case U'#':
      m_current = std::find_if(m_current, m_diagManager->getDocument().end(),
                               [](char32_t value) { return value == '\n'; });
      if (m_current == m_diagManager->getDocument().end())
        break;

      [[fallthrough]];
    case U'\n': {
      auto offset = m_current - m_diagManager->getDocument().begin();
      m_current++;
      if (m_depth == 0) {
        // Skip over blank logical lines. If the last token before the comment
        // was a newline, or we have no tokens, as this is file begin, no
        // newline tokens are generated because the whole lines was blank
        if (!isBlankLineSoFar(m_tokens))
          m_tokens.emplace_back(offset, 1, TokenType::Newline);

        parseIndent();
      }
      break;
    }
    case U'\\':
      m_current++;
      if (m_current == m_diagManager->getDocument().end()) {
        createError(m_current - m_diagManager->getDocument().begin(),
                    Diag::UNEXPECTED_EOF_WHILE_PARSING)
            .addHighlight(m_current - m_diagManager->getDocument().begin(),
                          "\\n", Diag::flags::insertColour);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - m_diagManager->getDocument().begin(),
                              TokenType::SyntaxError);
        return true;
      }
      if (*m_current != U'\n') {
        createError(
            m_current - m_diagManager->getDocument().begin(),
            Diag::UNEXPECTED_CHARACTER_AFTER_LINE_CONTINUATION_CHARACTER)
            .addHighlight(m_current - m_diagManager->getDocument().begin(),
                          "\\n", Diag::flags::insertColour,
                          Diag::flags::strikethrough);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - m_diagManager->getDocument().begin(),
                              TokenType::SyntaxError);
        return true;
      }
      m_current++;
      continue;

    case U'u':
    case U'U':
      m_current++;
      if (m_current != m_diagManager->getDocument().end() &&
          (*m_current == '\'' || *m_current == '"')) {
        if (auto opt = parseLiteral(false, false))
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::StringLiteral,
                                std::move(*opt));
        else
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::SyntaxError);

      } else {
        m_current--;
        parseIdentifier();
      }
      break;

    case U'\'':
    case U'"': {
      if (auto opt = parseLiteral(false, false))
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - start, TokenType::StringLiteral,
                              std::move(*opt));
      else
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - start, TokenType::SyntaxError);

      break;
    }
    case U'r':
    case U'R': {
      if (std::next(m_current) == m_diagManager->getDocument().end()) {
        parseIdentifier();
        break;
      }
      switch (*std::next(m_current)) {
        /*
      case U'f':
      case U'F': {
        auto maybeLiteral = std::next(m_current, 2);
        if (maybeLiteral == m_diagManager->getDocument().end() ||
            (*maybeLiteral != U'\'' && *maybeLiteral != U'"')) {
          parseIdentifier();
          break;
        }
        // TODO: parseFormatString
        std::abort();
      }
         */
      case U'\'':
      case U'"':
        m_current = std::next(m_current);
        if (auto opt = parseLiteral(true, false))
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::StringLiteral,
                                std::move(*opt));
        else
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::SyntaxError);

        break;

      case 'b':
      case 'B':
        if (std::next(m_current, 2) == m_diagManager->getDocument().end()) {
          parseIdentifier();
          break;
        }
        if (*std::next(m_current, 2) == '"' ||
            *std::next(m_current, 2) == '\'') {
          std::advance(m_current, 2);
        } else {
          parseIdentifier();
          break;
        }
        if (auto opt = parseLiteral(true, true))
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::ByteLiteral,
                                std::move(*opt));
        else
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                                m_current - start, TokenType::SyntaxError);

        break;
      default: parseIdentifier(); break;
      }
      break;
    }
    case U'b':
    case U'B': {
      if (std::next(m_current) == m_diagManager->getDocument().end()) {
        parseIdentifier();
        break;
      }
      bool raw = false;
      if (*std::next(m_current) == 'r' || *std::next(m_current) == 'R') {
        if (*std::next(m_current, 2) != '"' &&
            *std::next(m_current, 2) != '\'') {
          parseIdentifier();
          break;
        }
        raw = true;
        std::advance(m_current, 2);
      } else if (*std::next(m_current) == '"' ||
                 *std::next(m_current) == '\'') {
        m_current++;
      } else {
        parseIdentifier();
        break;
      }
      if (auto opt = parseLiteral(raw, true))
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - start, TokenType::ByteLiteral,
                              std::move(*opt));
      else
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                              m_current - start, TokenType::SyntaxError);

      break;
    }
      /*
    case U'f':
    case U'F': {
      if (std::next(m_current) != m_diagManager->getDocument().end() &&
          (*std::next(m_current) == '"' || *std::next(m_current) == '\'')) {
        // TODO: parse format string
        std::abort();
      } else {
        parseIdentifier();
      }
      break;
    }
       */
    case U'.':
      if (std::next(m_current) == m_diagManager->getDocument().end() ||
          *std::next(m_current) < U'0' || *std::next(m_current) > U'9') {
        m_current++;
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                              TokenType::Dot);
        break;
      }
      [[fallthrough]];

    case U'0':
    case U'1':
    case U'2':
    case U'3':
    case U'4':
    case U'5':
    case U'6':
    case U'7':
    case U'8':
    case U'9': parseNumber(); break;
    case U'-':
      if (std::next(m_current) != m_diagManager->getDocument().end() &&
          *std::next(m_current) == U'=') {
        std::advance(m_current, 2);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                              TokenType::MinusAssignment);
      } else if (std::next(m_current) != m_diagManager->getDocument().end() &&
                 *std::next(m_current) == U'>') {
        std::advance(m_current, 2);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                              TokenType::Arrow);
      } else {
        m_current++;
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                              TokenType::Minus);
      }
      break;

    case U'+':
    case U'%':
    case U'@':
    case U'&':
    case U'|':
    case U'^':
    case U':':
    case U'=': {
      auto [normal, assignment] = [&]() -> std::pair<TokenType, TokenType> {
        switch (*m_current) {
        case U'+': return {TokenType::Plus, TokenType::PlusAssignment};
        case U'%':
          return {TokenType::Remainder, TokenType::RemainderAssignment};
        case U'@': return {TokenType::AtSign, TokenType::AtAssignment};
        case U'&': return {TokenType::BitAnd, TokenType::BitAndAssignment};
        case U'|': return {TokenType::BitOr, TokenType::BitOrAssignment};
        case U'^': return {TokenType::BitXor, TokenType::BitXorAssignment};
        case U':': return {TokenType::Colon, TokenType::Walrus};
        case U'=': return {TokenType::Assignment, TokenType::Equal};
        default: PYLIR_UNREACHABLE;
        }
      }();
      if (std::next(m_current) == m_diagManager->getDocument().end() ||
          *std::next(m_current) != U'=') {
        m_current++;
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                              normal);
      } else if (std::next(m_current) != m_diagManager->getDocument().end()) {
        std::advance(m_current, 2);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                              assignment);
      }
      break;
    }
    case U'*':
    case U'/':
    case U'>':
    case U'<': {
      auto [single, singleAss, twice,
            twiceAss] = [&]() -> std::array<TokenType, 4> {
        switch (*m_current) {
        case U'*':
          return {TokenType::Star, TokenType::TimesAssignment,
                  TokenType::PowerOf, TokenType::PowerOfAssignment};
        case U'/':
          return {TokenType::Divide, TokenType::DivideAssignment,
                  TokenType::IntDivide, TokenType::IntDivideAssignment};
        case U'>':
          return {TokenType::GreaterThan, TokenType::GreaterOrEqual,
                  TokenType::ShiftRight, TokenType::ShiftRightAssignment};
        case U'<':
          return {TokenType::LessThan, TokenType::LessOrEqual,
                  TokenType::ShiftLeft, TokenType::ShiftLeftAssignment};
        default: PYLIR_UNREACHABLE;
        }
      }();
      if (std::next(m_current) != m_diagManager->getDocument().end() &&
          *std::next(m_current) == *m_current) {
        m_current++;
        if (std::next(m_current) != m_diagManager->getDocument().end() &&
            *std::next(m_current) == U'=') {
          std::advance(m_current, 2);
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 3,
                                twiceAss);
        } else {
          m_current++;
          m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                                twice);
        }
      } else if (std::next(m_current) != m_diagManager->getDocument().end() &&
                 *std::next(m_current) == U'=') {
        std::advance(m_current, 2);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                              singleAss);
      } else {
        m_current++;
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                              single);
      }
      break;
    }
    case U'~':
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::BitNegate);
      break;
    case U'(':
      m_depth++;
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::OpenParentheses);
      break;
    case U')':
      if (m_depth != 0)
        m_depth--;

      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::CloseParentheses);
      break;
    case U'[':
      m_depth++;
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::OpenSquareBracket);
      break;
    case U']':
      if (m_depth != 0)
        m_depth--;

      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::CloseSquareBracket);
      break;
    case U'{':
      m_depth++;
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::OpenBrace);
      break;
    case U'}':
      if (m_depth != 0)
        m_depth--;

      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::CloseBrace);
      break;
    case U',':
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::Comma);
      break;
    case U';':
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 1,
                            TokenType::SemiColon);
      break;
    case U'!':
      if (std::next(m_current) != m_diagManager->getDocument().end() &&
          *std::next(m_current) == U'=') {
        std::advance(m_current, 2);
        m_tokens.emplace_back(start - m_diagManager->getDocument().begin(), 2,
                              TokenType::NotEqual);
        break;
      }
      [[fallthrough]];
    default:
      if (Text::isWhitespace(*m_current)) {
        m_current = std::find_if_not(
            m_current, m_diagManager->getDocument().end(),
            [](char32_t c) { return Text::isWhitespace(c) && c != U'\n'; });
        continue;
      }
      parseIdentifier();
      break;
    }
  } while (m_current != m_diagManager->getDocument().end() &&
           startSize == m_tokens.size());
  if (m_current == m_diagManager->getDocument().end()) {
    m_tokens.emplace_back(m_current - m_diagManager->getDocument().begin(), 0,
                          TokenType::Newline);
    parseIndent();
  }
  return true;
}

void pylir::Lexer::parseIdentifier() {
  static auto initialCharacterSet =
      llvm::sys::UnicodeCharSet(INITIAL_CHARACTERS);
  if (!initialCharacterSet.contains(*m_current)) {
    createError(m_current - m_diagManager->getDocument().begin(),
                Diag::UNEXPECTED_CHARACTER_N,
                Text::toUTF8String({&*m_current, 1}))
        .addHighlight(m_current - m_diagManager->getDocument().begin());
    m_tokens.emplace_back(m_current - m_diagManager->getDocument().begin(), 1,
                          TokenType::SyntaxError);
    m_current++;
    return;
  }
  static auto legalIdentifierSet = llvm::sys::UnicodeCharSet(LEGAL_IDENTIFIERS);
  const auto* start = m_current;
  m_current = std::find_if_not(
      m_current, m_diagManager->getDocument().end(),
      [&](char32_t value) { return legalIdentifierSet.contains(value); });
  auto utf32 =
      std::u32string_view{start, static_cast<std::size_t>(m_current - start)};
  static std::unordered_map<std::u32string_view, TokenType> keywords = {
      {U"False", TokenType::FalseKeyword},
      {U"None", TokenType::NoneKeyword},
      {U"True", TokenType::TrueKeyword},
      {U"and", TokenType::AndKeyword},
      {U"as", TokenType::AsKeyword},
      {U"assert", TokenType::AssertKeyword},
      {U"async", TokenType::AsyncKeyword},
      {U"await", TokenType::AwaitKeyword},
      {U"break", TokenType::BreakKeyword},
      {U"class", TokenType::ClassKeyword},
      {U"continue", TokenType::ContinueKeyword},
      {U"def", TokenType::DefKeyword},
      {U"del", TokenType::DelKeyword},
      {U"elif", TokenType::ElifKeyword},
      {U"else", TokenType::ElseKeyword},
      {U"except", TokenType::ExceptKeyword},
      {U"finally", TokenType::FinallyKeyword},
      {U"for", TokenType::ForKeyword},
      {U"from", TokenType::FromKeyword},
      {U"global", TokenType::GlobalKeyword},
      {U"if", TokenType::IfKeyword},
      {U"import", TokenType::ImportKeyword},
      {U"in", TokenType::InKeyword},
      {U"is", TokenType::IsKeyword},
      {U"lambda", TokenType::LambdaKeyword},
      {U"nonlocal", TokenType::NonlocalKeyword},
      {U"not", TokenType::NotKeyword},
      {U"or", TokenType::OrKeyword},
      {U"pass", TokenType::PassKeyword},
      {U"raise", TokenType::RaiseKeyword},
      {U"return", TokenType::ReturnKeyword},
      {U"try", TokenType::TryKeyword},
      {U"while", TokenType::WhileKeyword},
      {U"with", TokenType::WithKeyword},
      {U"yield", TokenType::YieldKeyword},
  };
  if (auto result = keywords.find(utf32); result != keywords.end()) {
    m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                          m_current - start, result->second);
    return;
  }

  auto normalized = Text::normalize(utf32, Text::Normalization::NFKC);
  [[maybe_unused]] bool ok;
  auto utf8 = Text::toUTF8String(normalized, &ok);
  PYLIR_ASSERT(ok);
  m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                        m_current - start, TokenType::Identifier,
                        std::move(utf8));
}

namespace {
int fromHex(char32_t value) {
  switch (value) {
  case U'0': return 0;
  case U'1': return 1;
  case U'2': return 2;
  case U'3': return 3;
  case U'4': return 4;
  case U'5': return 5;
  case U'6': return 6;
  case U'7': return 7;
  case U'8': return 8;
  case U'9': return 9;
  case U'a':
  case U'A': return 10;
  case U'b':
  case U'B': return 11;
  case U'c':
  case U'C': return 12;
  case U'd':
  case U'D': return 13;
  case U'e':
  case U'E': return 14;
  case U'f':
  case U'F': return 15;
  default: PYLIR_UNREACHABLE;
  }
}

bool isHex(char32_t value) {
  switch (value) {
  case U'0':
  case U'1':
  case U'2':
  case U'3':
  case U'4':
  case U'5':
  case U'6':
  case U'7':
  case U'8':
  case U'9':
  case U'a':
  case U'b':
  case U'c':
  case U'd':
  case U'e':
  case U'f':
  case U'A':
  case U'B':
  case U'C':
  case U'D':
  case U'E':
  case U'F': return true;
  default: break;
  }
  return false;
}
} // namespace

std::optional<std::string> pylir::Lexer::parseLiteral(bool raw, bool bytes) {
  bool longString = false;
  auto character = *m_current++;
  if (m_current != m_diagManager->getDocument().end() &&
      std::next(m_current) != m_diagManager->getDocument().end())
    if (std::array{*m_current, *std::next(m_current)} ==
        std::array{character, character}) {
      longString = true;
      std::advance(m_current, 2);
    }

  auto end = [&]() -> std::optional<bool> {
    if (!longString) {
      if (m_current == m_diagManager->getDocument().end()) {
        createError(m_current - m_diagManager->getDocument().begin(),
                    Diag::EXPECTED_END_OF_LITERAL)
            .addHighlight(m_current - m_diagManager->getDocument().begin(),
                          std::string(1, character));
        return std::nullopt;
      }
      if (*m_current == character) {
        m_current++;
        return true;
      }
      return false;
    }
    if (m_current == m_diagManager->getDocument().end() ||
        std::next(m_current) == m_diagManager->getDocument().end() ||
        std::next(m_current, 2) == m_diagManager->getDocument().end()) {
      while (m_current != m_diagManager->getDocument().end())
        m_current = std::next(m_current);

      createError(m_current - m_diagManager->getDocument().begin(),
                  Diag::EXPECTED_END_OF_LITERAL)
          .addHighlight(m_current - m_diagManager->getDocument().begin(),
                        std::string(3, character));
      return std::nullopt;
    }
    if (std::array{*m_current, *std::next(m_current),
                   *std::next(m_current, 2)} ==
        std::array{character, character, character}) {
      std::advance(m_current, 3);
      return true;
    }
    return false;
  };
  std::u32string result;
  std::optional<bool> success;

  auto diagnoseNonAscii = [&] {
    auto builder =
        createError(m_current - m_diagManager->getDocument().begin(),
                    Diag::ONLY_ASCII_VALUES_ARE_ALLOWED_IN_BYTE_LITERALS)
            .addHighlight(m_current - m_diagManager->getDocument().begin());
    if (raw)
      return;

    std::string utf8Bytes = Text::toUTF8(*m_current).data();
    std::string hexEscape;
    for (auto iter : utf8Bytes) {
      hexEscape += fmt::format(
          FMT_STRING("\\x{:0^2X}"),
          static_cast<std::uint32_t>(static_cast<std::uint8_t>(iter)));
    }
    builder
        .addNote(m_current - m_diagManager->getDocument().begin(),
                 Diag::USE_HEX_OR_OCTAL_ESCAPES_INSTEAD)
        .addHighlight(m_current - m_diagManager->getDocument().begin(),
                      hexEscape, Diag::flags::insertColour,
                      Diag::flags::strikethrough);
  };

  for (success = end(); success && !*success; success = end()) {
    switch (*m_current) {
    case U'\\':
      if (raw) {
        result += *m_current;
        m_current++;
        break;
      }
      m_current++;
      if (m_current == m_diagManager->getDocument().end())
        continue;

      switch (*m_current) {
      case '\'':
      case '"':
      case '\\':
        result += *m_current;
        m_current++;
        break;
      case 'a':
      case 'b':
      case 'f':
      case 'r':
      case 't':
      case 'v': {
        char32_t escape;
        switch (*m_current) {
        case 'a': escape = '\a'; break;
        case 'b': escape = '\b'; break;
        case 'f': escape = '\f'; break;
        case 'r': escape = '\r'; break;
        case 't': escape = '\t'; break;
        case 'v': escape = '\v'; break;
        default: PYLIR_UNREACHABLE;
        }
        result += escape;
        m_current++;
        break;
      }
      case 'n': {
        m_current++;
        result += '\n';
        std::size_t count = 0;
        while (std::next(m_current, count) !=
                   m_diagManager->getDocument().end() &&
               count != sizeof("ewline") - 1)
          count++;

        if (std::u32string_view(std::prev(m_current), count + 1) == U"newline")
          std::advance(m_current, count);

        break;
      }
      case 'x': {
        m_current++;
        if (m_current == m_diagManager->getDocument().end()) {
          // TODO deprecation
          result += U"\\x";
          break;
        }
        if (!isHex(*m_current)) {
          // TODO deprecation
          result += U"\\x";
          result += *m_current;
          break;
        }
        m_current++;
        if (m_current == m_diagManager->getDocument().end()) {
          // TODO deprecation
          result += U"\\x";
          result += *std::prev(m_current);
          break;
        }
        if (!isHex(*m_current)) {
          // TODO deprecation
          result += U"\\x";
          result += *std::prev(m_current);
          result += *m_current;
          break;
        }
        char32_t unicode =
            fromHex(*std::prev(m_current)) * 16 + fromHex(*m_current);
        result += unicode;
        m_current++;
        break;
      }
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7': {
        char32_t value = 0;
        std::size_t count = 0;
        while (count < 3 && m_current != m_diagManager->getDocument().end() &&
               *m_current >= '0' && *m_current <= '7') {
          value = value * 8 + *m_current - U'0';
          m_current++;
          count++;
        }
        result += value;
        break;
      }
      case 'u':
      case 'U': {
        if (bytes) {
          result += U"\\";
          result += *m_current;
          m_current++;
          break;
        }
        bool big = *m_current == U'U';
        m_current++;
        std::size_t size = big ? 8 : 4;
        std::size_t count = 0;
        char32_t value = 0;
        while (count < size &&
               m_current != m_diagManager->getDocument().end() &&
               isHex(*m_current)) {
          value = value * 16 + fromHex(*m_current);
          m_current++;
          count++;
        }
        if (count != size) {
          createError(m_current - m_diagManager->getDocument().begin() - count,
                      Diag::EXPECTED_N_MORE_HEX_CHARACTERS, size - count)
              .addHighlight(
                  m_current - m_diagManager->getDocument().begin() - count,
                  m_current - m_diagManager->getDocument().begin() - 1)
              .addHighlight(
                  m_current - m_diagManager->getDocument().begin() - count - 2,
                  m_current - m_diagManager->getDocument().begin() - count - 1,
                  Diag::flags::secondaryColour);
          return std::nullopt;
        }
        if (!Text::isValidCodepoint(value)) {
          createError(m_current - m_diagManager->getDocument().begin() - count,
                      Diag::U_PLUS_N_IS_NOT_A_VALID_UNICODE_CODEPOINT,
                      static_cast<std::uint32_t>(value))
              .addHighlight(
                  m_current - m_diagManager->getDocument().begin() - count,
                  m_current - m_diagManager->getDocument().begin() - 1)
              .addHighlight(
                  m_current - m_diagManager->getDocument().begin() - count - 2,
                  m_current - m_diagManager->getDocument().begin() - count - 1,
                  Diag::flags::secondaryColour);
          return std::nullopt;
        }
        result += value;
        break;
      }
      case 'N': {
        if (bytes) {
          result += U"\\N";
          m_current++;
          break;
        }
        m_current++;
        if (m_current == m_diagManager->getDocument().end() ||
            *m_current != '{') {
          auto builder =
              createError(m_current - m_diagManager->getDocument().begin(),
                          Diag::EXPECTED_OPEN_BRACE_AFTER_BACKSLASH_N);
          if (m_current != m_diagManager->getDocument().end()) {
            builder.addHighlight(
                m_current - m_diagManager->getDocument().begin(), "{",
                Diag::flags::insertColour, Diag::flags::strikethrough);
          }
          builder.addHighlight(
              m_current - m_diagManager->getDocument().begin() - 2,
              m_current - m_diagManager->getDocument().begin() - 1,
              Diag::flags::secondaryColour);
          return std::nullopt;
        }
        m_current++;
        const auto* closing =
            std::find(m_current, m_diagManager->getDocument().end(), U'}');
        auto utf8Name = Text::toUTF8String(std::u32string_view{
            m_current, static_cast<std::size_t>(closing - m_current)});
        auto codepoint = Text::fromName(utf8Name);
        if (!codepoint) {
          auto builder =
              createError(m_current - m_diagManager->getDocument().begin(),
                          Diag::UNICODE_NAME_N_NOT_FOUND, utf8Name)
                  .addHighlight(
                      m_current - m_diagManager->getDocument().begin(),
                      closing - m_diagManager->getDocument().begin() - 1)
                  .addHighlight(
                      m_current - m_diagManager->getDocument().begin() - 2,
                      m_current - m_diagManager->getDocument().begin() - 1,
                      Diag::flags::secondaryColour);
          if (closing != m_diagManager->getDocument().end()) {
            builder.addHighlight(closing - m_diagManager->getDocument().begin(),
                                 Diag::flags::secondaryColour);
          }
          return std::nullopt;
        }
        if (closing != m_diagManager->getDocument().end())
          closing++;

        result += *codepoint;
        m_current = closing;
        break;
      }
      default: {
        if (bytes && *m_current > 127) {
          diagnoseNonAscii();
          return std::nullopt;
        }
        result += '\\';
        result += *m_current;
        m_current++;
        // TODO deprecation warning
        break;
      }
      }
      break;
    case U'\n': {
      if (!longString) {
        createError(m_current - m_diagManager->getDocument().begin(),
                    Diag::NEWLINE_NOT_ALLOWED_IN_LITERAL)
            .addHighlight(m_current - m_diagManager->getDocument().begin());
        return std::nullopt;
      }
      result += *m_current;
      m_current++;
      break;
    }
    default:
      if (bytes && *m_current > 127) {
        diagnoseNonAscii();
        return std::nullopt;
      }
      result += *m_current;
      m_current++;
      break;
    }
  }
  if (!success)
    return std::nullopt;

  if (bytes) {
    std::string byteConverted(result.begin(), result.end());
    return byteConverted;
  }
  return Text::toUTF8String(result);
}

void pylir::Lexer::parseNumber() {
  const auto* start = m_current;
  PYLIR_ASSERT(m_current != m_diagManager->getDocument().end());
  bool (*allowedDigits)(char32_t) =
      +[](char32_t value) { return value >= U'0' && value <= U'9'; };
  unsigned radix = 10;
  bool isFloat = false;
  if (*m_current == U'0' &&
      std::next(m_current) != m_diagManager->getDocument().end()) {
    switch (*std::next(m_current)) {
    case U'b':
    case U'B': {
      allowedDigits =
          +[](char32_t value) { return value == U'0' || value == U'1'; };
      std::advance(m_current, 2);
      radix = 2;
      break;
    }
    case U'o':
    case U'O': {
      allowedDigits =
          +[](char32_t value) { return value >= U'0' && value <= U'7'; };
      std::advance(m_current, 2);
      radix = 8;
      break;
    }
    case U'x':
    case U'X': {
      allowedDigits = isHex;
      std::advance(m_current, 2);
      radix = 16;
      break;
    }
    case U'0':
    case U'1':
    case U'2':
    case U'3':
    case U'4':
    case U'5':
    case U'6':
    case U'7':
    case U'8':
    case U'9':
    case U'e':
    case U'E':
    case U'.':
    default: break;
    }
  }
  const auto* numberStart = m_current;
  const auto* end =
      std::find_if_not(m_current, m_diagManager->getDocument().end(),
                       [allowedDigits, previous = U'\0', &isFloat,
                        radix](char32_t value) mutable {
                         if (value == U'.' && radix == 10) {
                           previous = U'.';
                           if (!isFloat) {
                             isFloat = true;
                             return true;
                           }
                           return false;
                         }
                         if (value == U'_') {
                           if (previous == U'_' || previous == U'.')
                             return false;

                           previous = value;
                           return true;
                         }
                         return allowedDigits(previous = value);
                       });
  m_current = end;
  if (*std::prev(end) == U'_') {
    createError(end - m_diagManager->getDocument().begin() - 1,
                Diag::UNDERSCORE_ONLY_ALLOWED_BETWEEN_DIGITS)
        .addHighlight(end - 1 - m_diagManager->getDocument().begin())
        .addHighlight(start - m_diagManager->getDocument().begin(),
                      end - m_diagManager->getDocument().begin() - 2,
                      Diag::flags::secondaryColour);
    m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                          end - start, TokenType::SyntaxError);
    return;
  }
  std::string text;
  for (auto codepoint : std::u32string_view{
           numberStart, static_cast<std::size_t>(end - numberStart)})
    if (codepoint != U'_')
      text += codepoint;

  auto checkSuffix = [&] {
    static auto legalIdentifierSet =
        llvm::sys::UnicodeCharSet(LEGAL_IDENTIFIERS);
    const auto* suffixEnd = std::find_if_not(
        m_current, m_diagManager->getDocument().end(),
        [&](char32_t value) { return legalIdentifierSet.contains(value); });
    if (suffixEnd == m_current)
      return;

    createError(m_current - m_diagManager->getDocument().begin(),
                Diag::INVALID_INTEGER_SUFFIX,
                Text::toUTF8String({m_current, static_cast<std::size_t>(
                                                   suffixEnd - m_current)}))
        .addHighlight(start - m_diagManager->getDocument().begin(),
                      m_current - m_diagManager->getDocument().begin() - 1,
                      Diag::flags::secondaryColour)
        .addHighlight(m_current - m_diagManager->getDocument().begin(),
                      suffixEnd - m_diagManager->getDocument().begin() - 1,
                      Diag::flags::strikethrough);
    m_tokens.emplace_back(m_current - m_diagManager->getDocument().begin(),
                          suffixEnd - m_current, TokenType::SyntaxError);
    m_current = suffixEnd;
  };
  isFloat = isFloat || (end != m_diagManager->getDocument().end() &&
                        (*end == U'e' || *end == U'E'));
  if (!isFloat) {
    BigInt integer(text, radix);
    if (radix == 10 && m_current != m_diagManager->getDocument().end() &&
        (*m_current == U'j' || *m_current == U'J')) {
      m_current++;
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                            m_current - start, TokenType::ComplexLiteral,
                            integer.roundToDouble());
      checkSuffix();
      return;
    }
    if (radix == 10 && !integer.isZero() && text.front() == '0') {
      const auto* leadingEnd =
          std::find_if_not(numberStart, end, [](char32_t value) {
            return value == U'_' || value == U'0';
          });

      createError(end - m_diagManager->getDocument().begin() - 1,
                  Diag::NUMBER_WITH_LEADING_ZEROS_NOT_ALLOWED)
          .addHighlight(leadingEnd - m_diagManager->getDocument().begin(),
                        end - m_diagManager->getDocument().begin() - 1,
                        Diag::flags::secondaryColour)
          .addHighlight(numberStart - m_diagManager->getDocument().begin(),
                        leadingEnd - m_diagManager->getDocument().begin() - 1)
          .addNote(numberStart - m_diagManager->getDocument().begin(),
                   Diag::REMOVE_LEADING_ZEROS)
          .addHighlight(numberStart - m_diagManager->getDocument().begin(),
                        leadingEnd - m_diagManager->getDocument().begin() - 1,
                        Diag::flags::secondaryColour,
                        Diag::flags::strikethrough);
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                            end - start, TokenType::SyntaxError);
      return;
    }
    m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                          end - start, TokenType::IntegerLiteral,
                          std::move(integer));
    checkSuffix();
    return;
  }

  if (text.front() == '.')
    text.insert(text.begin(), '0');

  if (end != m_diagManager->getDocument().end() &&
      (*end == U'e' || *end == U'E')) {
    text += U'e';
    end++;
    if (end != m_diagManager->getDocument().end() &&
        (*end == U'+' || *end == U'-')) {
      text += *end;
      end++;
    }
    const auto* newEnd = std::find_if_not(
        end, m_diagManager->getDocument().end(),
        [previous = U'\0', allowedDigits](char32_t value) mutable {
          if (value == U'_') {
            if (previous == U'_' || previous == U'\0') {
              return false;
            }
            previous = value;
            return true;
          }
          return allowedDigits(previous = value);
        });
    m_current = newEnd;
    if (newEnd == end) {
      createError(end - m_diagManager->getDocument().begin(),
                  Diag::EXPECTED_DIGITS_FOR_THE_EXPONENT)
          .addHighlight(start - m_diagManager->getDocument().begin(),
                        end - m_diagManager->getDocument().begin() - 1,
                        Diag::flags::secondaryColour)
          .addHighlight(end - m_diagManager->getDocument().begin());
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                            end - start, TokenType::SyntaxError);
      return;
    }

    for (auto codepoint :
         std::u32string_view{end, static_cast<std::size_t>(newEnd - end)})
      if (codepoint != U'_')
        text += codepoint;
  }

  double number;
#ifdef __cpp_lib_to_chars
  [[maybe_unused]] auto result =
      std::from_chars(text.data(), text.data() + text.size(), number);
  PYLIR_ASSERT(result.ec == std::errc());
#else
  struct LocalReset {
    std::locale locale{};

    LocalReset() = default;

    ~LocalReset() {
      std::locale::global(locale);
    }

    LocalReset(const LocalReset&) = delete;
    LocalReset& operator=(const LocalReset&) = delete;
    LocalReset(LocalReset&&) = delete;
    LocalReset& operator=(LocalReset&&) = delete;
  };
  std::optional<LocalReset> reset;
  if (std::use_facet<std::numpunct<char>>(std::locale()).decimal_point() !=
      '.') {
    reset.emplace();
    std::locale::global(std::locale::classic());
  }
  number = std::stod(std::string{text.begin(), text.begin() + text.size()});
  reset.reset();
#endif
  auto tokenType = TokenType::FloatingPointLiteral;
  if (m_current != m_diagManager->getDocument().end() &&
      (*m_current == U'j' || *m_current == U'J')) {
    m_current++;
    tokenType = TokenType::ComplexLiteral;
  }
  m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                        m_current - start, tokenType, number);
  checkSuffix();
}

void pylir::Lexer::parseIndent() {
  const auto* start = m_current;
  std::size_t indent = 0;
  for (; m_current != m_diagManager->getDocument().end() &&
         (Text::isWhitespace(*m_current) || *m_current == U'#');
       m_current++) {
    switch (*m_current) {
    case U'#':
    case U'\n': return;
    case U'\t':
      if (indent % 8 == 0)
        indent += 8;
      else
        indent = pylir::roundUpTo(indent, 8);

      break;
    default: indent++;
    }
  }
  if (indent < m_indentation.top().first) {
    std::pair<std::size_t, std::size_t> previous;
    do {
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                            start - m_diagManager->getDocument().begin(),
                            TokenType::Dedent);
      previous = m_indentation.top();
      m_indentation.pop();
    } while (indent < m_indentation.top().first);

    if (m_indentation.top().first != indent) {
      auto builder =
          createError(m_current - m_diagManager->getDocument().begin(),
                      Diag::INVALID_INDENTATION_N, indent)
              .addHighlight(start - m_diagManager->getDocument().begin(),
                            m_current - m_diagManager->getDocument().begin() -
                                1);
      if (previous.first - indent < indent - m_indentation.top().first) {
        builder
            .addNote(m_tokens[previous.second],
                     Diag::NEXT_CLOSEST_INDENTATION_N, previous.first)
            .addHighlight(m_tokens[previous.second]);
      } else if (m_indentation.top().first != 0) {
        builder
            .addNote(m_tokens[m_indentation.top().second],
                     Diag::NEXT_CLOSEST_INDENTATION_N,
                     m_indentation.top().first)
            .addHighlight(m_tokens[m_indentation.top().second]);
      }
      m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                            m_current - start, TokenType::SyntaxError);
    }
  } else if (indent > m_indentation.top().first) {
    m_tokens.emplace_back(start - m_diagManager->getDocument().begin(),
                          m_current - start, TokenType::Indent);
    m_indentation.emplace(indent, m_tokens.size() - 1);
  }
}
