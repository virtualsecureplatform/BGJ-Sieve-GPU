import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bundle_mpi_runtime', ROOT / 'tools' / 'bundle_mpi_runtime.py')
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class MpiHostRuntimeTests(unittest.TestCase):
    def test_bundle_copies_image_libraries_but_not_host_mpi(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            libraries = []
            for name in ('ld-linux-x86-64.so.2', 'libc.so.6', 'libcudart.so.13'):
                source = base / name
                source.write_bytes(('test-' + name).encode())
                libraries.append(source)
            output = ''.join('    %s => %s (0x1234)\n' % (p.name, p) for p in libraries)
            output += '    libmpi.so.40 => /opt/system/mpi/libmpi.so.40 (0x4567)\n'
            destination = base / 'runtime'
            with mock.patch.object(sys, 'argv', ['bundle', '/binary', str(destination)]), \
                 mock.patch.object(BUNDLE.subprocess, 'check_output', return_value=output):
                BUNDLE.main()
                with self.assertRaises(FileExistsError):
                    BUNDLE.main()
            manifest = json.loads((destination / 'manifest.json').read_text())
            self.assertEqual(len(manifest), 3)
            self.assertFalse((destination / 'libmpi.so.40').exists())
            for source in libraries:
                self.assertEqual(source.read_bytes(), (destination / source.name).read_bytes())

    def test_host_launcher_preserves_algorithm_environment_and_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            loader = runtime / 'ld-linux-x86-64.so.2'
            loader.write_text('#!%s\n' % sys.executable + '''
import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'spill':os.environ.get('HD_FAIL_ON_SPILL'),
                  'tls':os.environ.get('UCX_TLS'), 'unrelated':os.environ.get('UNRELATED_TEST')}))
''')
            loader.chmod(0o755)
            (runtime / 'libc.so.6').write_text('test library placeholder')
            env = dict(os.environ, MPI_HOST_RUNTIME=directory, MPI_BINARY='/test/binary',
                       MPI_SIF='/unused/image', MPI_BASE='/test/base', MPI_LIBDIR='/test/mpi',
                       APPTAINERENV_HD_FAIL_ON_SPILL='1', APPTAINERENV_UCX_TLS='rc',
                       APPTAINERENV_UNRELATED_TEST='not copied')
            env.pop('UNRELATED_TEST', None)
            result = subprocess.run(['bash', str(ROOT / 'reproduce' / 'svp167-search' / 'mpi_rank.sh'), 'argument'],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data['args'][0], '--library-path')
            self.assertTrue(data['args'][1].startswith(directory + ':'))
            self.assertEqual(data['args'][-2:], ['/test/binary', 'argument'])
            self.assertEqual(data['spill'], '1')
            self.assertEqual(data['tls'], 'rc')
            self.assertIsNone(data['unrelated'])


if __name__ == '__main__':
    unittest.main()
