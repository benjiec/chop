Cursor, please following these rules

1. When I ask you a question on why something is the way it is, do not assume I
want you to change it. Tell me if you want to change something and ask if that
is a change I want. Don't implement anything without a discussion. First
question should be "would you like me to propose a solution", rather than
"would you like me to implement". What would you implement w/o an agreed upon
approach?

2. Remember the purpose of a set of changes - and if that set of changes happen
to break something, the most important thing is NOT to fix what broke, but to
fix what broke AND preserve the purpose of the changes.

3. Don't change the configuration files and assume the code supports the syntax
changes in the configuration YAML files. Always check to make sure if you
add/modify the structure of the configuration YAML files, we have code that
supports the changes.

4. Do not produce Python code to just print statements that you will display in
rich text in the chat box. Avoid using the terminal to communicate messages -
only use it to run commands.

5. Prefer concise answers to simple questions, rather than expanding on an
answer with possibilities and providing lots of text to read.

6. Run unit tests using `scripts/run-unit-tests` whenever you update something
not in the tests/ directory.

Whenever we are done implement a set of logic and validated that they are what
we want, we should add unit tests and update the tests/unit/run_tests.py script
to include the new unit tests.

7. Don't leave comments that only pertain to an action you took, e.g. to undo
something. Comments should describe the current state of the code or
configuration.

8. Always use chop_env virtual env.
