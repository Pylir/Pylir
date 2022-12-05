#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import re
import sys


def main():
    if len(sys.argv) != 3:
        raise RuntimeError('Expected SOURCE and DEST arguments')

    [input_path, output_path] = sys.argv[1:]
    with open(input_path, 'r') as f:
        s = f.read()
        with open(output_path, 'w') as out:
            last_pos = 0
            for i in re.finditer('\\[TOC]', s):
                out.write(s[last_pos:i.start()])
                last_pos = i.end()
            out.write(s[last_pos:])


if __name__ == '__main__':
    main()
