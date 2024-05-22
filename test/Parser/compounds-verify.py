# RUN: pylir %s -fsyntax-only -verify

def test(default=5, **kwargs):
    pass

