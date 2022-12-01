#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import re
import sys
import textwrap


def main():
    if len(sys.argv) != 3:
        raise RuntimeError('Expected SOURCE and DEST arguments')

    [input_path, output_path] = sys.argv[1:]
    with open(input_path, 'r') as f:
        s = f.read()
        m = re.search('^#(.*)$', s, flags=re.MULTILINE)
        if not m:
            raise RuntimeError('Expected to find a title in Markdown document')
        title = m[1].lstrip()
        s = s[m.end():]
        with open(output_path, 'w') as out:
            format = textwrap.dedent(f'''\
            ---
            title: "{title}"
            date: 1970-01-01T00:00:00Z
            draft: false
            ---
            ''')
            out.write(format)
            last_pos = 0
            for i in re.finditer('\\[TOC]', s):
                out.write(s[last_pos:i.start()])
                out.write('<p/>{{< toc >}}')
                last_pos = i.end()
            out.write(s[last_pos:])


if __name__ == '__main__':
    main()
