"""Hot-loadable builtin tool plugins.

Drop a ``*.py`` file here that defines a LangChain ``BaseTool`` and calls
``app.tools.registry.register_builtin(...)``. The service scans this
directory at startup and at the beginning of every chat turn.
"""