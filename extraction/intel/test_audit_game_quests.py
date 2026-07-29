import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_game_quests.py")
SPEC = importlib.util.spec_from_file_location("audit_game_quests", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def quest_record(index: int) -> dict:
    return {
        "_id": f"{index:024x}",
        "traderId": "5" * 24,
        "conditions": {"AvailableForFinish": [{"id": f"{index + 1000:024x}"}]},
        "rewards": {"Success": [{"id": f"{index + 2000:024x}"}]},
    }


class CatalogClassificationTests(unittest.TestCase):
    def test_complete_large_catalog_is_accepted(self):
        doc = {"err": 0, "errmsg": None, "data": [quest_record(i) for i in range(150)]}
        result = audit.classify_catalog_document(doc, "LiveQuestTemplates")
        self.assertTrue(result["authoritative_production_catalog"])
        self.assertEqual(result["record_count"], 150)
        self.assertEqual(result["id_fraction"], 1.0)

    def test_test_named_catalog_is_rejected_even_when_structurally_complete(self):
        doc = {"data": [quest_record(i) for i in range(150)], "err": 0}
        result = audit.classify_catalog_document(doc, "TestQuestTemplates")
        self.assertFalse(result["authoritative_production_catalog"])
        self.assertIn("fixture_or_test_name", result["rejection_reasons"])

    def test_small_mock_is_rejected(self):
        result = audit.classify_catalog_document(
            [quest_record(1), quest_record(2)], "Quests"
        )
        self.assertFalse(result["authoritative_production_catalog"])
        self.assertIn("record_count<100", result["rejection_reasons"])

    def test_profile_progress_is_not_mistaken_for_catalog(self):
        doc = {
            "Quests": [
                {
                    "qid": "1" * 24,
                    "status": "Success",
                    "completedConditions": ["2" * 24],
                }
            ]
        }
        result = audit.classify_catalog_document(doc, "BigProfile")
        self.assertFalse(result["authoritative_production_catalog"])


class EvidenceTests(unittest.TestCase):
    def test_locale_coverage_compares_names_and_objectives(self):
        task_a, task_b, objective = "1" * 24, "2" * 24, "3" * 24
        baseline = {
            task_a: {"id": task_a, "name": "Exact"},
            task_b: {"id": task_b, "name": "Different"},
        }
        locale = {
            "err": 0,
            "data": {
                f"{task_a} name": "Exact",
                f"{task_b} name": "Changed",
                f"{objective} description": "Objective",
            },
        }
        result = audit.locale_coverage(locale, baseline, {objective})
        self.assertEqual(result["baseline_task_names_present"], 2)
        self.assertEqual(result["baseline_task_names_exact"], 1)
        self.assertEqual(result["baseline_objective_ids_present_as_locale_key"], 1)

    def test_profile_evidence_links_source_and_condition_ids(self):
        task, condition = "a" * 24, "b" * 24
        doc = {
            "Quests": [{"qid": task, "completedConditions": [condition]}],
            "TaskConditionCounters": {
                condition: {
                    "id": condition,
                    "sourceId": task,
                    "type": "HandoverItem",
                }
            },
        }
        result = audit.profile_evidence(doc, {task}, {condition})
        self.assertEqual(result["task_ids_matching_baseline"], 1)
        self.assertEqual(result["condition_ids_matching_baseline"], 1)
        self.assertEqual(result["condition_type_histogram"], {"HandoverItem": 1})

    def test_dump_parser_extracts_json_fields_and_offsets(self):
        sample = """
// Namespace: EFT.Quests
public class QuestTemplate // TypeDefIndex: 1
{
    // Fields
    [JsonProperty("_id")]
    private string <Id>k__BackingField; // 0x10
    [JsonProperty("conditions")]
    private ConditionsDict <Conditions>k__BackingField; // 0x20

    // Properties
}
"""
        classes = audit.parse_dump_classes(sample, ("QuestTemplate",))
        self.assertIn("QuestTemplate", classes)
        self.assertEqual(classes["QuestTemplate"]["fields"][0]["json_property"], "_id")
        self.assertEqual(classes["QuestTemplate"]["fields"][1]["offset"], "0x20")


if __name__ == "__main__":
    unittest.main()
