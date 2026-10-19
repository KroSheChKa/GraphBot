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
class TestDeriveBackendExtra(unittest.TestCase):
    def test_basic(self):
        x = symbols('x')
        cases = [('x^2',1,'2*x'), ('x^3',1,'3*x^2'), ('sin(x)',1,'cos(x)'), ('cos(x)',1,'-sin(x)'), ('exp(x)',1,'exp(x)'), ('ln(x)',1,'1/x')]
        for expr, order, exp in cases:
            f = sympify(expr)
            df = f
            for _ in range(order):
                df = diff(df, x)
            js = to_js(df)
            self.assertTrue(exp in js or js == exp)

if __name__ == '__main__':
    unittest.main()
