from gemma_swe.tasks import Task, is_test_path, parse_patch

PATCH = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -10,4 +10,6 @@ def f():
     a = 1
-    b = 2
+    b = 3
+    c = 4
+    d = 5
     return a
@@ -40 +42,2 @@ def g():
-    x
+    y
+    z
--- /dev/null
+++ b/pkg/new_mod.py
@@ -0,0 +1,2 @@
+VALUE = 1
+OTHER = 2
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 import pkg
+assert pkg
"""


def test_parse_patch_counts_and_ranges():
    core, new_mod, test = parse_patch(PATCH)
    assert (core.path, core.added, core.removed, core.is_new) == ("pkg/core.py", 5, 2, False)
    assert core.old_ranges == ((10, 13), (40, 40))
    assert (new_mod.path, new_mod.added, new_mod.is_new) == ("pkg/new_mod.py", 2, True)
    assert new_mod.old_ranges == ()
    assert test.path == "tests/test_core.py"


def test_parse_patch_accepts_git_headers():
    git_style = (
        "diff --git a/x.py b/x.py\nindex 1..2 100644\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    (only,) = parse_patch(git_style)
    assert (only.path, only.added, only.removed) == ("x.py", 1, 1)


def test_is_test_path_matches_harness_rules():
    assert is_test_path("tests/test_core.py")
    assert is_test_path("pkg/core_test.py")
    assert is_test_path("testing/helpers.py")
    assert not is_test_path("pkg/core.py")
    assert not is_test_path("tests/data/fixture.json")
    assert not is_test_path("pkg/testing_utils.py")


def test_gold_source_paths_exclude_tests():
    task = Task("x_1", "o/x", "0" * 40, "issue", patch=PATCH)
    assert task.gold_source_paths == ["pkg/core.py", "pkg/new_mod.py"]
    assert task.graph_stem() == "x_" + "0" * 40
