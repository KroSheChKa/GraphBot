import unittest
try:
    from sympy import symbols, sympify, diff
    from symengine import sympify as se_sympify, Symbol as se_Symbol, diff as se_diff
    _HAS = True
except Exception:
    _HAS = False

def to_js(s):
    t = str(s)
    t = t.replace('**', '^')
    t = t.replace('Abs(', 'abs(')
    t = t.replace('log(', 'ln(')
    return t

@unittest.skipUnless(_HAS, 'requires sympy/symengine')
class TestDeriveBackend(unittest.TestCase):
    def test_basic(self):
        x = symbols('x')
        f = sympify('x^2')
        df = diff(f, x)
        self.assertEqual(to_js(df), '2*x')

