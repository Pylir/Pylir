# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck --match-full-lines %s

import foo
import foo as bar
import foo, bar

# CHECK: |-import
# CHECK-NEXT: | `-module foo
# CHECK-NEXT: |-import
# CHECK-NEXT: | `-as bar: module foo
# CHECK-NEXT: |-import
# CHECK-NEXT: | |-module foo
# CHECK-NEXT: | `-module bar

import foo.bar

# CHECK-NEXT: |-import
# CHECK-NEXT: | `-module foo.bar

from foo import bar
from foo import bar as foobar
from foo import (bar as foobar)
from foo import (bar as foobar, )

# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module
# CHECK-NEXT: | | `-module foo
# CHECK-NEXT: | `-bar
# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module
# CHECK-NEXT: | | `-module foo
# CHECK-NEXT: | `-bar as foobar
# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module
# CHECK-NEXT: | | `-module foo
# CHECK-NEXT: | `-bar as foobar
# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module
# CHECK-NEXT: | | `-module foo
# CHECK-NEXT: | `-bar as foobar

from . import bar
from .. import bar
from .foo import bar

# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module .
# CHECK-NEXT: | `-bar
# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module ..
# CHECK-NEXT: | `-bar
# CHECK-NEXT: |-import list
# CHECK-NEXT: | |-relative module .
# CHECK-NEXT: | | `-module foo
# CHECK-NEXT: | `-bar

from __future__ import division
from __future__ import division as foo, annotations
from __future__ import (division as foo, annotations)
from __future__ import (division as foo, annotations, )

# CHECK-NEXT: |-import futures
# CHECK-NEXT: | `-division
# CHECK-NEXT: |-import futures
# CHECK-NEXT: | |-division as foo
# CHECK-NEXT: | `-annotations
# CHECK-NEXT: |-import futures
# CHECK-NEXT: | |-division as foo
# CHECK-NEXT: | `-annotations
# CHECK-NEXT: `-import futures
# CHECK-NEXT: |-division as foo
# CHECK-NEXT: `-annotations
