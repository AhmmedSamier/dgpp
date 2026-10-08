"""Prevent incomplete or invalid measurements from entering the targeted rerun."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import result_sources as sources


class ResultSourcesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.here = self.root / 'new'
        self.old = self.root / 'old/raw/model/refresh/default'
        self.new = self.here / 'raw/model/refresh/default'
        replacement = patch.object(sources, 'HERE', self.here)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.write(self.here / 'manifest.json', {'binary_sha256': 'new-binary'})
        self.write(self.here / 'configs/model.json', {'model': 'GLM Flash', 'nodes': 4})
        self.install(self.old, 'old-binary', [2048, 8192, 32768])
        self.bind_old()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def install(self, directory, binary, lengths):
        for name, value in {
            'manifest.json': {'binary_sha256': binary},
            'memory-check.json': {'passed': True},
            'rank-identity.json': {'passed': True},
            'down.log.command.json': {'returncode': 0},
            'prefill.log.command.json': {'returncode': 0},
            'prefill.json': {'samples': [
                {'requested_tokens': n, 'repeat': r, 'value': binary}
                for n in lengths for r in range(3)]},
            'context-0.log.command.json': {'returncode': 0},
            'context-0.json': {'samples': [{'value': binary}]},
            'context-32768.log.command.json': {'returncode': 0},
            'context-32768.json': {'samples': [{'value': binary}]},
        }.items():
            self.write(directory / name, value)

    def bind_old(self):
        self.write(self.here / 'reused-results.json', {
            'source_binary_sha256': 'old-binary',
            'modes': {'model/default': {
                'directory': '../old/raw/model/refresh/default',
                'target_config': {'path': 'configs/model.json', 'sha256': hashlib.sha256(
                    (self.here / 'configs/model.json').read_bytes()).hexdigest()},
                'jobs': {'prefill': {'lengths': [2048, 8192]}, 'context-0': {}},
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in self.old.glob('*.json')},
            }},
        })

    def test_old_long_prefill_and_long_context_cannot_be_reused(self):
        self.assertEqual(sources.prefill_missing_lengths(self.new), [32768])
        self.assertIsNone(sources.completed(self.new, 'prefill'))
        self.assertIsNone(sources.completed(self.new, 'context-32768'))
        self.assertTrue(sources.completed(self.new, 'context-0')['_sources'][0]['reused'])
        partial = sources.prefill_report(self.new, require_complete=False)
        self.assertEqual({s['requested_tokens'] for s in partial['samples']}, {2048, 8192})

    def test_targeted_prefill_combines_only_allowed_lengths_with_exact_provenance(self):
        self.install(self.new, 'new-binary', [32768])
        report = sources.completed(self.new, 'prefill')
        self.assertEqual(len(report['samples']), 9)
        self.assertEqual([(s['reused'], s['lengths']) for s in report['_sources']],
                         [(True, [2048, 8192]), (False, [32768])])
        self.assertTrue(all(s['value'] == ('new-binary' if s['requested_tokens'] == 32768
                                         else 'old-binary') for s in report['samples']))

    def test_complete_fresh_results_supersede_reuse(self):
        self.install(self.new, 'new-binary', [2048, 8192, 32768])
        report = sources.completed(self.new, 'prefill')
        self.assertEqual(len(report['_sources']), 1)
        self.assertFalse(report['_sources'][0]['reused'])
        self.assertTrue(all(s['value'] == 'new-binary' for s in report['samples']))

    def test_failed_fresh_launch_cannot_contribute_even_completed_jobs(self):
        for filename, value in [
            ('memory-check.json', {'passed': False}),
            ('rank-identity.json', {'passed': False}),
            ('down.log.command.json', {'returncode': 1}),
            ('prefill.log.command.json', {'returncode': 1}),
            ('manifest.json', {'binary_sha256': 'unexpected-binary'}),
        ]:
            with self.subTest(filename=filename):
                self.install(self.new, 'new-binary', [32768])
                self.write(self.new / filename, value)
                self.assertIsNone(sources.completed(self.new, 'prefill'))
                self.assertEqual(sources.prefill_missing_lengths(self.new), [32768])

    def test_missing_memory_evidence_cannot_contribute(self):
        self.install(self.new, 'new-binary', [32768])
        (self.new / 'memory-check.json').unlink()
        self.assertIsNone(sources.completed(self.new, 'prefill'))

    def test_changed_audited_source_fails_closed(self):
        self.write(self.old / 'memory-check.json', {'passed': False})
        with self.assertRaisesRegex(RuntimeError, 'reused evidence changed'):
            sources.completed(self.new, 'context-0')

    def test_changed_deployment_cannot_reuse_old_results(self):
        self.write(self.here / 'configs/model.json', {'model': 'GLM Flash', 'nodes': 2})
        with self.assertRaisesRegex(RuntimeError, 'target configuration changed'):
            sources.completed(self.new, 'context-0')

    def test_hash_match_alone_cannot_authorize_failed_old_launch(self):
        self.write(self.old / 'memory-check.json', {'passed': False})
        self.bind_old()
        with self.assertRaisesRegex(RuntimeError, 'no longer passes validation'):
            sources.completed(self.new, 'context-0')

    def test_partial_fresh_length_keeps_old_samples_and_attribution(self):
        self.install(self.new, 'new-binary', [8192, 32768])
        report = sources.read(self.new / 'prefill.json')
        report['samples'] = [s for s in report['samples']
                             if not (s['requested_tokens'] == 8192 and s['repeat'] == 2)]
        self.write(self.new / 'prefill.json', report)
        result = sources.completed(self.new, 'prefill')
        self.assertEqual([(s['reused'], s['lengths']) for s in result['_sources']],
                         [(True, [2048, 8192]), (False, [32768])])

    def test_duplicate_repetitions_do_not_count_as_complete(self):
        self.install(self.new, 'new-binary', [32768])
        report = sources.read(self.new / 'prefill.json')
        report['samples'][-1]['repeat'] = 1
        self.write(self.new / 'prefill.json', report)
        self.assertIsNone(sources.completed(self.new, 'prefill'))

    def bind_retained(self, lengths):
        directory = self.here / 'archived/default'
        self.install(directory, 'intermediate-binary', [2048, 8192, 32768])
        catalog = sources.read(self.here / 'reused-results.json')
        catalog['source_binary_sha256'] = 'intermediate-binary'
        entry = catalog['modes']['model/default']
        entry['directory'] = 'archived/default'
        entry['jobs'] = {'prefill': {'lengths': lengths}, 'context-0': {}}
        entry['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in directory.glob('*.json')}
        self.write(self.here / 'retained-results.json', catalog)
        return directory

    def test_three_binary_generations_keep_exact_per_length_sources(self):
        self.bind_retained([8192])
        self.install(self.new, 'new-binary', [32768])
        report = sources.completed(self.new, 'prefill')
        self.assertEqual([(s['binary_sha256'], s['lengths']) for s in report['_sources']],
                         [('old-binary', [2048]), ('intermediate-binary', [8192]),
                          ('new-binary', [32768])])
        self.assertEqual([s['value'] for s in report['samples']],
                         ['old-binary'] * 3 + ['intermediate-binary'] * 3 + ['new-binary'] * 3)

    def test_retained_allowance_does_not_restore_affected_lengths(self):
        (self.here / 'reused-results.json').unlink()
        self.bind_old()
        self.bind_retained([2048])
        (self.here / 'reused-results.json').unlink()
        self.assertEqual(sources.prefill_missing_lengths(self.new), [8192, 32768])
        self.assertIsNone(sources.completed(self.new, 'context-32768'))

    def test_changed_retained_evidence_fails_closed(self):
        directory = self.bind_retained([2048])
        self.write(directory / 'memory-check.json', {'passed': False})
        with self.assertRaisesRegex(RuntimeError, 'reused evidence changed'):
            sources.completed(self.new, 'context-0')

    def test_new_binary_supersedes_both_retention_catalogs(self):
        self.bind_retained([2048, 8192, 32768])
        self.install(self.new, 'new-binary', [2048, 8192, 32768])
        report = sources.completed(self.new, 'prefill')
        self.assertEqual(len(report['_sources']), 1)
        self.assertEqual(report['_sources'][0]['binary_sha256'], 'new-binary')


if __name__ == '__main__':
    unittest.main()
