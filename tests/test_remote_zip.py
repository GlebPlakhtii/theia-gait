import io
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from select_remote_zip import acquire, RangeReader


class RemoteZipTests(unittest.TestCase):
    def fixture(self):
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('selected/video.mp4',b'example media payload')
            z.writestr('unselected.txt',b'not selected')
        return f.getvalue()

    def opener(self,data,ignored=False):
        def op(req,timeout=None):
            a,b=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',req.get_header('Range')).groups())
            class Response(io.BytesIO):
                status=200 if ignored else 206
                headers={'Content-Range':f'bytes {a}-{b}/{len(data)}','ETag':'stable'}
            return Response(data[a:b+1])
        return op

    def test_selective_acquisition_and_crc(self):
        data=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch('urllib.request.urlopen',self.opener(data)):
            d=Path(d)
            r=acquire('https://example.test/archive',len(data),d/'audit.json',['selected/video.mp4'],d/'out')
            self.assertEqual((d/'out/video.mp4').read_bytes(),b'example media payload')
            self.assertFalse((d/'out/unselected.txt').exists())
            self.assertTrue(r['selected_files'][0]['crc32_verified'])
            self.assertFalse(r['full_archive_sha256_verified'])

    def test_ignored_ranges_and_budget_rejected(self):
        data=self.fixture()
        with patch('urllib.request.urlopen',self.opener(data,True)):
            with self.assertRaises(ValueError):RangeReader('https://example.test/archive',len(data)).read(20)
        with self.assertRaises(ValueError):RangeReader('https://example.test/archive',len(data),budget=5).read(20)


if __name__=='__main__':unittest.main()
