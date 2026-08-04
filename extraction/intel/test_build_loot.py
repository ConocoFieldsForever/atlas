import unittest

import build_loot


class MapVariantSelectionTests(unittest.TestCase):
    def test_factory_day_record_wins_in_either_catalog_order(self):
        day = {'normalizedName': 'factory', 'name': 'Factory'}
        night = {'normalizedName': 'night-factory', 'name': 'Night Factory'}

        for records in ([day, night], [night, day]):
            selected = build_loot.select_map_variants(records)
            self.assertEqual([record['name'] for record in selected], ['Factory'])

    def test_unmapped_records_are_ignored(self):
        selected = build_loot.select_map_variants([
            {'normalizedName': 'unknown-map', 'name': 'Unknown'},
            {'normalizedName': 'woods', 'name': 'Woods'},
        ])
        self.assertEqual([record['name'] for record in selected], ['Woods'])


if __name__ == '__main__':
    unittest.main()
