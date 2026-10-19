import unittest

try:
    import sympy  # noqa: F401
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

try:
    import flask  # noqa: F401
    _HAS_FLASK = True
except Exception:
    _HAS_FLASK = False


@unittest.skipUnless(_HAS_SYMPY and _HAS_FLASK, "requires sympy and flask")
class DeriveServerTests(unittest.TestCase):
    def setUp(self):
        from tools.approximator_server import app

        self.client = app.test_client()

    def post(self, expr, order):
        return self.client.post("/api/derive", json={"expr": expr, "order": order})

    def test_identity(self):
        r = self.post("x", 0)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(r.get_json()["formula"], "x")

    def test_derivatives(self):
        cases = [
            ("x^2", 1, "2*x"),
            ("x^3", 2, "6*x"),
            ("sin(x)", 1, "cos(x)"),
        ]
        for expr, order, expected in cases:
            with self.subTest(expr=expr, order=order):
                r = self.post(expr, order)
                self.assertEqual(r.status_code, 200)
                body = r.get_json()
                self.assertTrue(body["ok"], body)
                self.assertIn(expected, body["formula"])

    def test_order_clamping(self):
        r = self.post("x^2", 10)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(r.get_json()["formula"], "0")

    def test_negative_order_clamped_to_zero(self):
        r = self.post("x^2", -5)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(r.get_json()["formula"], "x^2")

    def test_empty_expr(self):
        r = self.post("", 1)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("ok", body)

    def test_to_js_formula_helper(self):
        from tools.approximator_server import _to_js_formula
        self.assertEqual(_to_js_formula(None), "")
        self.assertEqual(_to_js_formula("x**2"), "x^2")


if __name__ == "__main__":
    unittest.main()
