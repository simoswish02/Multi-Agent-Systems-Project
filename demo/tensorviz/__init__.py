"""Drawing the network's actual inputs.

Everything under ``demo/tensorviz`` renders arrays taken straight out of
``obs["global"]`` / ``obs["local"]`` and ``ctx_to_numpy`` -- never a
hand-drawn illustration of them. ``verify.py`` asserts that what is drawn
matches what the environment and the network compute, and the build fails if
it does not.
"""
