#Define set of molecules for use in RPAtest
molecules = {
    "H2": {
        "atom": "H 0 0 0; H 0 0 2.00",
        "unit": "Bohr", "spin": 0, "symmetry": False},

    "H2O": {
        "atom": """
        O  0.000000  0.000000  0.117300
        H  0.000000  0.757200 -0.469200
        H  0.000000 -0.757200 -0.469200
        """,
        "unit": "Angstrom", "spin": 0, "symmetry": False}
}