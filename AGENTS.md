Cursor, please following these rules

1. When answering a question, don't assume you need to make changes
immediately. Propose solution, get confirmation, then make the change.

2. Don't fix the a bug just to get rid of the error, but remember the purpose
of the changes made that may have caused the bug and preserve that purpose.

3. Always check to make sure if you add/modify the structure of the
configuration YAML files, we have code that supports the changes. Don't assume
the code is there.

4. Do not produce Python code to just print statements in the terminal, when
you can display the statements in the chat box. Avoid using terminal to
communicate messages with user.

5. Prefer concise answers to simple questions, rather than expanding on an
answer with possibilities and providing lots of text to read.

6. Run unit tests using `scripts/run-unit-tests` whenever you update something
not in the tests/ directory.

7. Whenever adding code outside of the tests/ directory, consider adding unit
tests to tests/ directory and tests/run_tests.py driver.

8. Don't leave comments that only pertain to an action you took, e.g. to undo
something. Comments should describe the current state of the code or
configuration.

9. Follow DRY model in code development. Abstract common code into functions
that can be re-called as much as possible.

10. Always use chop_env virtual env.

11. Whenever possible, use constants in utils/constants.py. Specifically,
always use the GenePredictionClass enums from that module to refer to training
target values, and DNAClass enums to refer to embeddings.
