#!/usr/bin/env python3
"""Exercise cleanup refusal paths without touching real cache files or SSH."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cleanup_generated_resident as cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.root = self.home / '.cache/dgpp/resident'
        self.root.mkdir(parents=True)
        self.name = '0123456789abcdef.img'
        self.image = self.root / self.name
        self.image.write_bytes(b'new cache')
        self.original = self.root / 'fedcba9876543210.img'
        self.original.write_bytes(b'original cache')

    def invoke(self, *, remove=True, initial=None, result=None):
        request = {'images': [self.name], 'initial': initial or [self.original.name], 'remove': remove}
        result = result or subprocess.CompletedProcess([], 1, '', '')
        output = io.StringIO()
        with patch.object(Path, 'home', return_value=self.home), \
             patch('sys.argv', ['cleanup', json.dumps(request)]), \
             patch('subprocess.run', return_value=result) as run, \
             contextlib.redirect_stdout(output):
            exec(compile(cleanup.REMOTE, '<remote-cleanup-test>', 'exec'), {})
            if not remove:
                run.assert_not_called()
        return json.loads(output.getvalue())

    def test_dry_run_preserves_files(self):
        record = self.invoke(remove=False)
        self.assertEqual(len(record['images']), 1)
        self.assertEqual(record['removed'], [])
        self.assertTrue(self.image.exists())

    def test_remove_only_new_unused_image_and_digest(self):
        self.image.with_suffix('.digest').write_bytes(b'new digest')
        record = self.invoke()
        self.assertEqual(len(record['removed']), 2)
        self.assertFalse(self.image.exists())
        self.assertFalse(self.image.with_suffix('.digest').exists())
        self.assertEqual(self.original.read_bytes(), b'original cache')
        self.assertEqual(self.invoke()['removed'], [])  # Repeat is harmless.

    def test_original_image_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'Original cache'):
            self.invoke(initial=[self.name])
        self.assertTrue(self.image.exists())

    def test_original_digest_is_rejected_before_any_deletion(self):
        self.image.with_suffix('.digest').write_bytes(b'original digest')
        with self.assertRaisesRegex(AssertionError, 'Original cache'):
            self.invoke(initial=[self.image.with_suffix('.digest').name])
        self.assertTrue(self.image.exists())

    def test_symlink_is_rejected_and_target_preserved(self):
        self.image.unlink()
        self.image.symlink_to(self.original)
        with self.assertRaises(AssertionError):
            self.invoke()
        self.assertTrue(self.image.is_symlink())
        self.assertEqual(self.original.read_bytes(), b'original cache')

    def test_active_cache_and_failed_inspection_are_rejected(self):
        for result in [subprocess.CompletedProcess([], 0, '123\n', ''),
                       subprocess.CompletedProcess([], 1, '', 'permission denied')]:
            with self.subTest(result=result), self.assertRaisesRegex(AssertionError, 'in use or inspection failed'):
                self.invoke(result=result)
            self.assertTrue(self.image.exists())

    def test_incomplete_model_cannot_start_remote_cleanup(self):
        model = json.loads((cleanup.HERE / 'matrix.json').read_text())[0]['model']
        with patch('sys.argv', ['cleanup', '--model', model, '--remove']), \
             patch.object(cleanup, 'entry_complete', return_value=False), \
             patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'outstanding measurements'):
                cleanup.main()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
