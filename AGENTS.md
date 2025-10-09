LLM, please following these rules

1. When answering a question, don't assume you need to make changes
immediately. Propose solution, get confirmation, then make the change.

2. Don't fix the a bug just to get rid of the error, but remember the purpose
of the changes made that may have caused the bug and preserve that purpose.

3. Do not produce Python code to just print statements in the terminal, when
you can display the statements in the chat box. Avoid using terminal to
communicate messages with user.

4. Prefer concise answers to simple questions, rather than expanding on an
answer with possibilities and providing lots of text to read.

5. Run unit tests using `scripts/run-unit-tests` whenever you update something
not in the tests/ directory.

6. Please add new unit tests whenever you are adding new features. New tests
should follow Python unittest module convention (e.g. subclass from
unittest.TestCase). IMPORTANT - place all tests for a .py file into the same
test .py file, i.e. don't put different behavior tests for the same .py file
into multiple test files.

7. Don't leave comments that only pertain to an action you took, e.g. to undo
something. Comments should describe the current state of the code or
configuration.

8. Follow DRY model in code development. Abstract common code into functions
that can be re-called as much as possible.

9. DO NOT seed the random number generator in code. Put random seeding in unit
tests instead to get consistent behavior.

10. Always use chop_env virtual env. For example, running python should be like
```source chop_env/bin/activate && python ...```

11. Whenever possible, use constants in utils/constants.py. Specifically,
always use the GenePredictionClass enums from that module to refer to training
target values, and DNAEmbed enums to refer to embeddings. If you need embedding
integer to DNA base pair mapping, use utils.constants.DNAEmbed.idx_to_bp. If
you need class index to class name mapping, use
utils.constants.GenePredictionClass.idx_to_cls. Don't roll your own
dictionaries.

12. When removing old code we are not using, which you should almost always do,
please remove all the code including the function signature/definitino, and
don't just leave an empty "pass" or "return None" as a placeholder. And remove
tests for logic we don't use anymore. And, DO NOT leave a comment saying you
removed something.

13. Use these metrics and terminology.
Sensitivity = TP / (TP + FN)
Precision = TP / (TP + FP)
Specificity = TN / (TN + FP)

15. Do not use try/except unless catching a specific, documented exception
type. Never use bare except or catch broad types like Exception/BaseException.
For example, please DO NOT use "except:" or "except Exception:".

16. Do not use single line functions that just returns outcome of another
function with the exact same arguments. Just go and change all the use of
former to the latter.

17. Do not leave empty functions around if they are not used anymore. E.g. "def
hello(): pass" or "def hello(): return". Just remove code we don't use.

18. Do not use this style "getattr(self, <field name>, None)" - just set the
field in the class and use self.<field name>.
