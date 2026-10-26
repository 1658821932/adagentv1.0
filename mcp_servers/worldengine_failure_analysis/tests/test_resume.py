import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import prepare_attempt
from evidence import save
from scripts.watch_batch import counts


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_attempt_and_stops_at_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=prepare_attempt(root,'token',True,2)
            save(first/'status.json',{'state':'failed'});save(first/'raw.json',{'original':True})
            second=prepare_attempt(root,'token',True,2)
            self.assertNotEqual(first,second)
            self.assertEqual((root/'scenes/token').resolve(),second.resolve())
            self.assertTrue(json.loads((first/'raw.json').read_text())['original'])
            save(second/'status.json',{'state':'failed'})
            self.assertIsNone(prepare_attempt(root,'token',True,2))
            self.assertEqual(counts(root,['token','pending']),{'total':2,'completed':0,'failed':1,'running':0,'pending':1})

    def test_completed_scene_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);first=prepare_attempt(root,'token',True,3)
            save(first/'status.json',{'state':'completed'})
            self.assertIsNone(prepare_attempt(root,'token',True,3))
            self.assertEqual(len(list((root/'attempts/token').iterdir())),1)
            with self.assertRaises(FileExistsError):prepare_attempt(root,'token',False,3)


if __name__=='__main__':unittest.main()
