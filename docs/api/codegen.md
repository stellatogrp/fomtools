# Code generation

OSQP-codegen-style generation of low-level solvers for *verified* first-order
methods, plus a tiny Python wrapper. An LLM coding agent translates each step of
the algorithm into the target language and iterates against a Python reference
oracle until the generated code is numerically verified.

::: fomtools.codegen.generate_code

::: fomtools.codegen.CodegenConfig

::: fomtools.codegen.extract_ir

::: fomtools.codegen.FomIR

::: fomtools.codegen.ReferenceOracle

::: fomtools.codegen.AgentDriver

::: fomtools.codegen.StubAgentDriver
