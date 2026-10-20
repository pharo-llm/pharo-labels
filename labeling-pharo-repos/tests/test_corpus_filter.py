"""The study-corpus filter: separate criteria, a funnel, and a diversity choice only above the cap."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharo_task_miner.config import DEFAULTS  # noqa: E402
from pharo_task_miner.sampling.corpus_filter import CRITERIA, failed_criteria, requirement, select, study_repo_ids  # noqa: E402
from pharo_task_miner.storage.database import Database  # noqa: E402


def repo(i=0, **over):
    m = {"repo_id": i, "full_name": f"owner{i}/repo{i}", "owner": f"owner{i}", "license": "MIT", "stars": 0, "forks": 0,
         "contributors": 3, "commit_count": 200, "primary_domain": "web_development", "size_category": "medium",
         "project_kind": "framework_or_library", "pushed_at": "2026-01-01T00:00:00Z", "created_at": "2022-01-01T00:00:00Z",
         "development_days": 1461, "years_since_last_push": 0.8, "classes": 40, "baselines": 1, "tonel_files": 45,
         "methods": 600, "test_methods": 150, "non_test_methods": 450, "test_method_share": 0.25,
         "duplicated_method_share": 0.01, "parse_failure_share": 0.0}
    m.update(over)
    return m


class CorpusFilterTest(unittest.TestCase):
    def setUp(self):
        self.f = dict(DEFAULTS["corpus_filter"])

    def test_a_good_repository_passes(self):
        self.assertEqual(failed_criteria(repo(), self.f), [])

    def test_each_criterion_excludes_on_its_own(self):
        bad = {"license": {"license": None}, "contributors": {"contributors": 1}, "commits": {"commit_count": 49},
               "development_days": {"development_days": 30}, "last_push": {"years_since_last_push": 6.0},
               "baseline": {"baselines": 0}, "tonel": {"tonel_files": 0}, "non_test_methods": {"non_test_methods": 199},
               "max_methods": {"methods": 20001}, "test_methods": {"test_methods": 19},
               "test_share": {"test_method_share": 0.09}, "duplication": {"duplicated_method_share": 0.11},
               "parse_failures": {"parse_failure_share": 0.02}}
        self.assertEqual(set(bad), {c[0] for c in CRITERIA})
        for key, change in bad.items():
            self.assertEqual(failed_criteria(repo(**change), self.f), [key], key)

    def test_unknown_licence_and_missing_measurements_fail(self):
        self.assertEqual(failed_criteria(repo(license="NOASSERTION"), self.f), ["license"])
        self.assertEqual(failed_criteria(repo(contributors=None), self.f), ["contributors"])

    def test_a_criterion_set_to_zero_is_off(self):
        f = {**self.f, "min_commits": 0, "require_license": False}
        self.assertEqual(failed_criteria(repo(commit_count=1, license=None), f), [])
        self.assertEqual(requirement("commits", f), "off")

    def test_funnel_counts_and_first_failed_reason(self):
        rows = [repo(1), repo(2, license=None, contributors=1), repo(3, contributors=1)]
        decisions, funnel, info = select(rows, self.f)
        by = {d["repo_id"]: d for d in decisions}
        self.assertEqual(by[1]["decision"], "included")
        self.assertEqual(by[2]["reason"], "a licence that GitHub recognises")
        self.assertEqual(by[2]["failed"], ["license", "contributors"])
        steps = {r["criterion"]: r for r in funnel}
        self.assertEqual((steps["license"]["removed"], steps["license"]["remaining"]), (1, 2))
        self.assertEqual((steps["contributors"]["fail_alone"], steps["contributors"]["removed"]), (2, 1))
        self.assertEqual(funnel[-1]["remaining"], info["selected"])
        self.assertFalse(info["capped"])

    def test_above_the_cap_the_choice_is_for_diversity_and_repeatable(self):
        rows = [repo(i, primary_domain="web_development" if i < 18 else f"domain{i}", stars=1000 - i) for i in range(20)]
        f = {**self.f, "max_repositories": 5}
        decisions, _, info = select(rows, f)
        chosen = [d for d in decisions if d["decision"] == "included"]
        self.assertEqual((len(chosen), info["capped"], info["passing"]), (5, True, 20))
        self.assertTrue({"domain18", "domain19"} <= {d["primary_domain"] for d in chosen})  # rare domains are covered
        self.assertEqual(sum(d["decision"] == "not_selected" for d in decisions), 15)
        again = [d["repo_id"] for d in select(rows, f)[0] if d["decision"] == "included"]
        self.assertEqual(again, [d["repo_id"] for d in chosen])

    def test_no_filter_run_means_no_restriction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Database(Path(tmp) / "db.sqlite") as db:
                self.assertIsNone(study_repo_ids(db))


if __name__ == "__main__":
    unittest.main()
