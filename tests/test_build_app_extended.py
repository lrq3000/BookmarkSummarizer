
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import subprocess

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import build_app

class TestBuildApp(unittest.TestCase):

    @patch('build_app.subprocess.check_call')
    def test_install_pyinstaller(self, mock_check_call):
        # Case 1: PyInstaller imported successfully
        with patch.dict(sys.modules, {'PyInstaller': MagicMock()}):
            build_app.install_pyinstaller()
            mock_check_call.assert_not_called()

        # Case 2: PyInstaller not imported (simulated by Import Error)
        # To simulate import error properly we use a side_effect on __import__
        original_import = __import__
        def side_effect(name, *args, **kwargs):
            if name == 'PyInstaller':
                raise ImportError("No module named PyInstaller")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=side_effect):
            # We also need to make sure it's not in sys.modules
            with patch.dict(sys.modules):
                if 'PyInstaller' in sys.modules:
                    del sys.modules['PyInstaller']

                build_app.install_pyinstaller()
                mock_check_call.assert_called_with([sys.executable, "-m", "pip", "install", "pyinstaller"])

    @patch('build_app.subprocess.check_call')
    @patch('build_app.shutil.copy2')
    @patch('os.path.exists')
    def test_build_executable(self, mock_exists, mock_copy2, mock_check_call):
        # Mock existence of files
        mock_exists.return_value = True # For all exists checks

        # Capture stdout
        with patch('sys.stdout', new=MagicMock()):
            build_app.build_executable()

        self.assertEqual(mock_check_call.call_count, 3) # 3 scripts
        self.assertTrue(mock_copy2.called) # copy config

    @patch('build_app.subprocess.check_call')
    @patch('build_app.shutil.copy2')
    @patch('os.path.exists')
    def test_build_executable_add_data_missing(self, mock_exists, mock_copy2, mock_check_call):
        # Simulate add-data source missing
        # The script checks `if os.path.exists(src):` for add_data items.
        # We need to make it return False for custom_parsers but True for other things if needed?
        # Actually it only checks for items in `add_data` list.
        # We can just return False.
        mock_exists.return_value = False

        with patch('sys.stdout', new=MagicMock()):
            build_app.build_executable()

        # Should still run build commands, just without add-data args
        self.assertEqual(mock_check_call.call_count, 3)
        # Check that --add-data was NOT in the args for crawl (which has add_data)
        # crawl is the second call
        args, _ = mock_check_call.call_args_list[1]
        cmd_list = args[0]
        self.assertNotIn("--add-data", cmd_list)

    @patch('build_app.subprocess.check_call')
    def test_build_executable_failure(self, mock_check_call):
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "cmd")

        with self.assertRaises(SystemExit):
            build_app.build_executable()

if __name__ == '__main__':
    unittest.main()
