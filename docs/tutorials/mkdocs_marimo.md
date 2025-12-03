/// marimo-embed
    height: 400px
    mode: read
    app_width: wide

    ```python
    @app.cell
    def __():
        import marimo as mo
        name = mo.ui.text(placeholder="Enter your name", debounce=False)
        name
        return

    @app.cell
    def __():
        mo.md(f"Hello, **{name.value or '__'}**!")
        return
    ```
///
