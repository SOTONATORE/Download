"""Тесты: параллельные исправления без главных батчей и конфликт рта у одного персонажа.

Русский: тесты написаны в стиле unittest (работают и под pytest, и под `python -m unittest`).
Модель и сеть не вызываются: пакет исправления подменяется заглушкой, которая ждёт барьер на
три пакета. При последовательном исполнении барьер не сходится и тест падает.
"""
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# корень репозитория (папка, где лежит Generate/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Generate.core import gemini_prompts as gp  # noqa: E402


class _FakeBuilder:
    """Заглушка построителя: батчей нет, нужные номера и конфиг — как у настоящего."""

    def __init__(self, segments, wanted_nums, cfg):
        self.wanted = sorted(set(wanted_nums))
        self.cfg = cfg
        self.texts = {}
        self.by_segment = {}
        self.system = ""
        self.schema = {}

    def batches(self, nums):
        return []


def _seed(num):
    return gp.PromptResult(num=num, prompt="", prompt_raw=f"prompt {num}", rework_pending=["x"])


class ParallelFixWithoutMainBatchesTest(unittest.TestCase):

    def test_fix_packages_run_concurrently_without_main_batches(self):
        # 6 клипов при fix_batch_size=2 — 3 пакета исправления, главных батчей нет
        cfg = gp.PromptConfig(style_file="unused.md", media="image", fix_batch_size=2)
        seeds = [_seed(n) for n in range(1, 7)]
        barrier = threading.Barrier(3, timeout=5)   # все три пакета должны быть в работе одновременно
        lock = threading.Lock()
        names = []
        active = {"now": 0, "max": 0}

        def stub_process_fix(self, logged, clips, fix_no):
            with lock:
                names.append(threading.current_thread().name)
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            try:
                barrier.wait()   # при последовательном исполнении: BrokenBarrierError по таймауту
            finally:
                with lock:
                    active["now"] -= 1
            with self.out_lock:
                for c in clips:
                    self.out[c.num] = c

        with mock.patch.object(gp, "_Builder", _FakeBuilder), \
                mock.patch.object(gp, "preset_text", return_value=""), \
                mock.patch.object(gp, "_provider_key_count", return_value=10), \
                mock.patch.object(gp._WaveJob, "process_fix", stub_process_fix):
            results = gp.generate_prompts(
                [], [], cfg, transport=object(), model="m", max_workers=10, provider="openrouter",
                rework=seeds)

        self.assertEqual([r.num for r in results], [1, 2, 3, 4, 5, 6])
        self.assertGreater(len(set(names)), 1)
        self.assertEqual(active["max"], 3)
        self.assertTrue(all(n.startswith("gemini-fix-") for n in names))


class SharedPoolPerWaveCapTest(unittest.TestCase):

    def _job(self, rework, batches=()):
        builder = SimpleNamespace(cfg=SimpleNamespace(fix_batch_size=2))
        return gp._WaveJob("фото", builder, list(batches), None, "m", "", None, tag="", rework=rework)

    def test_zero_batches_with_rework_gives_more_than_one_thread(self):
        pool = gp.SharedBatchPool({"фото": 10}, per_wave_cap=True)
        job = self._job([_seed(n) for n in range(1, 31)])   # 30 клипов / 2 = 15 пакетов
        self.assertEqual(pool.arrive_and_wait("фото", job), 10)

    def test_small_rework_caps_threads_by_package_count(self):
        pool = gp.SharedBatchPool({"фото": 10}, per_wave_cap=True)
        job = self._job([_seed(n) for n in range(1, 5)])    # 4 клипа / 2 = 2 пакета
        self.assertEqual(pool.arrive_and_wait("фото", job), 2)

    def test_no_work_keeps_one_thread(self):
        pool = gp.SharedBatchPool({"фото": 10}, per_wave_cap=True)
        job = self._job([])
        self.assertEqual(pool.arrive_and_wait("фото", job), 1)


class MouthConflictTest(unittest.TestCase):

    def test_single_farmer_closed_line_mouth_and_smile_is_conflict(self):   # клипы 331 и 342
        text = ("A farmer stands in a wheat field with a closed-line mouth. "
                "The farmer nods, showing a closed-line smile.")
        self.assertTrue(gp._mouth_conflict(text))

    def test_single_farmer_closed_line_mouth_and_worried_mouth_is_conflict(self):
        text = ("A farmer stands by a fence with a closed-line mouth. "
                "The farmer frowns with a worried wavy mouth.")
        self.assertTrue(gp._mouth_conflict(text))

    def test_single_farmer_repeated_same_mouth_is_not_conflict(self):   # клип 675
        text = ("A farmer holds a basket with a closed-line smile. "
                "The farmer nods, a closed-line smile on the face.")
        self.assertFalse(gp._mouth_conflict(text))

    def test_two_distinct_characters_in_separate_sentences_is_not_conflict(self):   # клип 641
        text = ("The main character has a round head and a closed-line smile. "
                "An adult figure stands beside, with a worried wavy mouth.")
        self.assertFalse(gp._mouth_conflict(text))

    def test_two_conflicting_mouths_in_same_single_character_clause_is_conflict(self):
        text = "The main character has a closed-line mouth and a worried wavy mouth."
        self.assertTrue(gp._mouth_conflict(text))

    def test_two_characters_each_with_one_mouth_in_separate_clauses_is_not_conflict(self):
        text = "A farmer with a closed-line smile and an adult figure with a worried wavy mouth."
        self.assertFalse(gp._mouth_conflict(text))

    def test_adjacent_words_of_one_noun_phrase_are_one_character(self):
        self.assertFalse(gp._has_multiple_characters("An adult figure smiles."))
        self.assertFalse(gp._has_multiple_characters("A farmer waves. The farmer smiles."))
        self.assertTrue(gp._has_multiple_characters("A farmer waves. An adult figure smiles."))


if __name__ == "__main__":
    unittest.main()
