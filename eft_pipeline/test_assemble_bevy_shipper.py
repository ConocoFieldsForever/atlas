import os
import tempfile
import unittest
from unittest import mock

from eft_pipeline.assemble_bevy import _PackShipper


class PackShipperTests(unittest.TestCase):
    def test_cross_volume_copy_retries_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            source = os.path.join(source_dir, 'texture.png')
            payload = (b'atlas-texture' * 1024) + b'end'
            with open(source, 'wb') as handle:
                handle.write(payload)

            real_copyfile = __import__('shutil').copyfile
            attempts = 0

            def flaky_copy(src, dst):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    with open(dst, 'wb') as handle:
                        handle.write(payload[:4096])
                    raise PermissionError('simulated interrupted SMB write')
                return real_copyfile(src, dst)

            shipper = _PackShipper(out_dir)
            with mock.patch('eft_pipeline.assemble_bevy.os.link', side_effect=OSError('cross-volume')):
                with mock.patch('eft_pipeline.assemble_bevy.shutil.copyfile', side_effect=flaky_copy):
                    relative = shipper.ship(source, 'tex/texture.png')

            destination = os.path.join(out_dir, 'tex', 'texture.png')
            self.assertEqual(relative, 'tex/texture.png')
            self.assertEqual(attempts, 2)
            with open(destination, 'rb') as handle:
                self.assertEqual(handle.read(), payload)
            self.assertFalse(any('.copying-' in name for name in os.listdir(os.path.dirname(destination))))


if __name__ == '__main__':
    unittest.main()
