import unittest
import tempfile
import os
import json
from src.ingest.pre_processor import config

class TestPreProcessorConfig(unittest.TestCase):
    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            f.write(json.dumps({"input_dir": ".", "output_dir": "."}))
            f.flush()
            cfg = config.load_config(f.name)
            self.assertEqual(str(cfg.input_dir), ".")
            self.assertEqual(str(cfg.output_dir), ".")
        os.unlink(f.name)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            config.load_config("/no/such/file.json")

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            f.write("not json")
            f.flush()
            with self.assertRaises(json.JSONDecodeError):
                config.load_config(f.name)
        os.unlink(f.name)

    def test_save_config_and_permissions(self):
        cfg = config.load_config(config.__file__.replace("config.py", "config_example.json"))
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            path = f.name
        try:
            config.save_config(cfg, path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

if __name__ == '__main__':
    unittest.main()
