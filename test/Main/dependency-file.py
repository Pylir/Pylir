# RUN: split-file %s %t
# RUN: pylir %t/main.py -emit-pylir -o %t/main.exe -M - | FileCheck %s

#--- main.py

import foo.bar.baz

#--- foo/__init__.py

#--- foo/bar/__init__.py

#--- foo/bar/baz.py

print(True)

# CHECK: main.exe:
# CHECK-DAG: builtins.py
# CHECK-DAG: foo{{/|\\}}__init__.py
# CHECK-DAG: foo{{/|\\}}bar{{/|\\}}__init__.py
# CHECK-DAG: foo{{/|\\}}bar{{/|\\}}baz.py
