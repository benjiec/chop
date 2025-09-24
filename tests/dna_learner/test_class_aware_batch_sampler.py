import unittest

from dna_learner.samplers import ClassAwareBatchSampler


class TestClassAwareBatchSampler(unittest.TestCase):
    def test_covers_target_classes_per_batch_when_available(self):
        # indices 0..9, classsets:
        class_map = {
            0: {1},
            1: {2},
            2: set(),
            3: {1},
            4: {2},
            5: {1, 2},
            6: set(),
            7: {1},
            8: {2},
            9: set(),
        }

        def index_to_classset(i: int):
            return class_map.get(int(i), set())

        indices = list(range(10))
        sampler = ClassAwareBatchSampler(
            indices=indices,
            batch_size=4,
            target_class_ids=[1, 2],
            index_to_classset=index_to_classset,
            seed=123,
            drop_last=False,
        )

        batches = list(iter(sampler))
        self.assertGreaterEqual(len(batches), 1)
        # In the first few batches, both classes should be represented (given availability)
        for b in batches[:2]:
            classes_in_batch = set()
            for idx in b:
                classes_in_batch.update(index_to_classset(idx))
            self.assertIn(1, classes_in_batch)
            self.assertIn(2, classes_in_batch)

    def test_deterministic_with_seed(self):
        class_map = {
            0: {1}, 1: {2}, 2: set(), 3: {1}, 4: {2}, 5: {1, 2}, 6: set(), 7: {1}, 8: {2}, 9: set()
        }

        def index_to_classset(i: int):
            return class_map.get(int(i), set())

        indices = list(range(10))
        sampler1 = ClassAwareBatchSampler(indices, 3, [1, 2], index_to_classset, seed=42)
        sampler2 = ClassAwareBatchSampler(indices, 3, [1, 2], index_to_classset, seed=42)
        b1 = list(iter(sampler1))
        b2 = list(iter(sampler2))
        self.assertEqual(b1, b2)

    def test_handles_insufficient_class_coverage(self):
        # Only class 1 appears; class 2 never appears
        class_map = {0: {1}, 1: set(), 2: {1}, 3: set()}

        def index_to_classset(i: int):
            return class_map.get(int(i), set())

        sampler = ClassAwareBatchSampler(indices=[0, 1, 2, 3], batch_size=2,
                                         target_class_ids=[1, 2], index_to_classset=index_to_classset, seed=7)
        batches = list(iter(sampler))
        self.assertGreaterEqual(len(batches), 2)
        # Ensure no duplicates within a batch and indices are valid
        for b in batches:
            self.assertEqual(len(b), len(set(b)))
            for idx in b:
                self.assertIn(idx, [0, 1, 2, 3])

    def test_single_multiclass_item_satisfies_multiple_requirements(self):
        # Only item 5 provides class 1; it also provides class 2.
        # With batch_size=1 and target classes [1,2], picking 5 once should satisfy both.
        class_map = {
            0: {2},
            1: set(),
            2: set(),
            3: set(),
            4: set(),
            5: {1, 2},  # multi-class and sole provider of class 1
        }

        def index_to_classset(i: int):
            return class_map.get(int(i), set())

        indices = list(range(6))
        sampler = ClassAwareBatchSampler(
            indices=indices,
            batch_size=1,
            target_class_ids=[1, 2],
            index_to_classset=index_to_classset,
            seed=99,
            drop_last=False,
        )

        batches = list(iter(sampler))
        self.assertGreaterEqual(len(batches), 1)
        b0 = batches[0]
        self.assertEqual(len(b0), 1)
        self.assertIn(5, b0)
        # And both classes are satisfied within the batch's items
        classes_in_b0 = set()
        for idx in b0:
            classes_in_b0.update(index_to_classset(idx))
        self.assertIn(1, classes_in_b0)
        self.assertIn(2, classes_in_b0)


if __name__ == '__main__':
    unittest.main()


