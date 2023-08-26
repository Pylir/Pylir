#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
import re
from functools import reduce
from typing import Callable

TOC_REC_REGEX = '\\[TOC]'
INTERFACE_IMPL_NODE = '^\\s*NOTE: This method \\*must\\* be implemented by the user\\.\\s*$'
TITLE_REC_REGEX = '^#.*$'
STRIP_TITLE_REC_REGEX = '^#\\s.*'


class RegexReplaceFile:
    def __init__(self, output_path, input_string):
        self.output_path = output_path
        self.input_string = input_string
        self.actions = []

    def add_rule(self, regex: str, action: Callable[[str], str]):
        self.actions += [(regex, action)]

    def run(self):
        reg = reduce(lambda curr, r: curr + f'|({r[0]})', self.actions[1:], f'({self.actions[0][0]})') if len(
            self.actions) > 0 else ''

        p = Path(self.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w') as out:
            last_pos = 0
            for i in re.finditer(reg, self.input_string, flags=re.MULTILINE):
                out.write(self.input_string[last_pos:i.start()])
                for index, j in enumerate(i.groups()):
                    if j is None:
                        continue
                    out.write(self.actions[index][1](j))

                last_pos = i.end()
            out.write(self.input_string[last_pos:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('dest')
    parser.add_argument('-title-indent', type=int, metavar='N', default=None)
    parser.add_argument('-strip-title', action='store_true')

    args = parser.parse_args()
    input_path, output_path = args.source, args.dest
    with open(input_path, 'r') as f:
        s = f.read()
        r = RegexReplaceFile(output_path, s)
        r.add_rule(TOC_REC_REGEX, lambda _: '')
        r.add_rule(INTERFACE_IMPL_NODE, lambda _: '')
        if args.strip_title:
            r.add_rule(STRIP_TITLE_REC_REGEX, lambda _: '')
        if args.title_indent is not None:
            r.add_rule(TITLE_REC_REGEX, lambda t: '#' * args.title_indent + t)
        r.run()


if __name__ == '__main__':
    main()
