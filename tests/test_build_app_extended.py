
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile

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

        # Case 2: PyInstaller not imported (simulated by KeyError)
        # We need to simulate ImportError when importing PyInstaller.
        # Since we can't easily modify the import mechanism for a specific module without affecting others or using builtins patch,
        # and builtins patch is risky/complex to scope correctly for just one import statement inside a function.
        # However, we can mock sys.executable and subprocess.check_call to verify the installation command is called if import fails.
        # But how to trigger import failure?
        # If we remove 'PyInstaller' from sys.modules, Python will try to find it.
        # If it finds it in site-packages, it imports it.
        # We want it to NOT find it.

        # We can use side_effect on __import__ but it's global.
        # Alternatively, we can assume the environment has PyInstaller (as evidenced by Case 1 working with mock)
        # and skip this test case or accept it's hard to test import failure without a virtualenv sandbox manipulation.

        # But wait, if we are in a test, we can use `unittest.mock.patch('builtins.__import__')` with a side effect
        # that raises ImportError only for 'PyInstaller'.

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

                # Now run the function
                build_app.install_pyinstaller()

                # Verify check_call was called with pip install
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
    def test_build_executable_failure(self, mock_check_call):
        mock_check_call.side_effect = build_app.subprocess.CalledProcessError(1, "cmd")

        with self.assertRaises(SystemExit):
            build_app.build_executable()

if __name__ == '__main__':
    unittest.main()
