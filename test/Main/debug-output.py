# Horrible hack to make sure all 'loc's are 'loc(#loc)'
# RUN: pylir %s -emit-pylir -o - -S | %python -c 'import re; import sys; sys.exit(0 if re.search("loc\\((?<!#loc)\\)", sys.stdin.read()) is None else 1)'
# RUN: pylir %s -emit-pylir -o - -S -g0 | %python -c 'import re; import sys; sys.exit(0 if re.search("loc\\((?<!#loc)\\)", sys.stdin.read()) is None else 1)'

# RUN: pylir %s -emit-pylir -o - -S -g | FileCheck %s --check-prefix=DEBUG

# DEBUG: loc({{.*debug-output\.py.*}})
