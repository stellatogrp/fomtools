/// marimo-embed
    height: 800 px
    mode: edit
    app_width: wide

    ```python 
    @app.cell
    def _():
        import matplotlib.pyplot as plt
        import numpy as np

        from algoverify.verifier import Verifier

        np.set_printoptions(precision=5)  # Print few decimal places
        np.set_printoptions(suppress=True)  # Suppress scientific notation
        return Verifier, np, plt
    ```
///
