#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class GlobInclude(Directive):
    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Element]:
        document = self.state.document
        source_dir = Path(document["source"]).absolute().parent
        pattern = self.arguments[0]

        content = []

        for path in sorted(source_dir.glob(pattern)):
            if len(content) > 0:
                content += ['</br>', '']
            content += ['```{include} ' + str(path.relative_to(source_dir)), '```']

        for idx, line in enumerate(content):
            self.content.data.insert(idx, line)
            self.content.items.insert(idx, (None, idx))

        node = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, node)
        return node.children


def setup(app: Sphinx):
    app.add_directive('globinclude', GlobInclude)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
