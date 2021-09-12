from collections import defaultdict

import sortedcontainers
import sys
import unicodedata

unicode_category = defaultdict(sortedcontainers.SortedList)
for c in map(chr, range(sys.maxunicode + 1)):
    unicode_category[unicodedata.category(c)].add(c)

id_start = unicode_category['Lu'] + unicode_category['Ll'] + unicode_category[
    'Lt'] + unicode_category['Lm'] + unicode_category['Lo'] + unicode_category[
               'Nl'] + unicode_category['Other_ID_Start'] + ['_']

id_continue = sortedcontainers.SortedSet(
    id_start + unicode_category['Mn'] + unicode_category['Mc'] +
    unicode_category['Nd'] + unicode_category['Pc'] +
    unicode_category['Other_ID_Continue'])

xid_continue = sortedcontainers.SortedSet([c for c in id_continue if
                                           unicodedata.normalize('NFKC',
                                                                 c) in id_continue])

xid_start = sortedcontainers.SortedSet()
for c in id_start:
    normalized = unicodedata.normalize('NFKC', c)
    if normalized[0] in id_start and all(
            c in xid_continue for c in normalized[1:]):
        xid_start.add(c)


def gen_unicode_char_range(iterable, variable_name):
    def finish_range(list_of_two):
        print(
            '{' + hex(ord(list_of_two[0])) + ', ' + hex(ord(
                list_of_two[1])) + '},')

    print("constexpr llvm::sys::UnicodeCharRange " + variable_name + "[] = {")

    current_range = None
    for c in iterable:
        if current_range is None:
            current_range = [c, c]
            continue

        if ord(current_range[1]) + 1 == ord(c):
            current_range[1] = c
        else:
            finish_range(current_range)
            current_range = [c, c]

    print("};")


gen_unicode_char_range(xid_start, "initialCharacters")
gen_unicode_char_range(xid_continue, "legalIdentifiers")
