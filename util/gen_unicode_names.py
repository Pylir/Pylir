#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests

UNICODE_VERSION_STR = "13.0.0"


def download_unicode_txt_file(name: str):
    r = requests.get(
        f'https://www.unicode.org/Public/{UNICODE_VERSION_STR}/ucd/{name}',
        allow_redirects=True)
    return r.content.decode('utf-8')


def unicode_text_file(content: str):
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line.split(';')


unicode_map = {}
for fields in unicode_text_file(download_unicode_txt_file('UnicodeData.txt')):
    codepoint = fields[0]
    name = fields[1]
    if name.startswith('<') and name.endswith('>'):
        continue
    if '..' not in codepoint:
        value = int(codepoint, 16)
    else:
        start = codepoint.split('..')[0]
        value = int(start, 16)
    unicode_map[name] = value

for fields in unicode_text_file(download_unicode_txt_file('NameAliases.txt')):
    name = fields[1]
    codepoint = fields[0]
    unicode_map[name] = int(codepoint, 16)

print("static const char* NAME_DATA[] = {")
for key, value in unicode_map.items():
    print(f'"{key}",', sep='')
print("};")
print("")
print("static char32_t NAME_CODEPOINT[] = {")
for key, value in unicode_map.items():
    print(f'{hex(value)},', sep='')
print("};")
