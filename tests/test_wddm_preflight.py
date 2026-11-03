import unittest
from personal_llm_data.personal_runtime import wddm_desktop_exclusions


def xml(mode='WDDM', kind='C+G', name='Code.exe'):
    return f'<nvidia_smi_log><gpu><driver_model><current_dm>{mode}</current_dm></driver_model><processes><process_info><pid>10</pid><type>{kind}</type><process_name>{name}</process_name></process_info></processes></gpu></nvidia_smi_log>'


class WddmPreflightTests(unittest.TestCase):
    def test_live_desktop_confirmed_but_python_or_reused_pid_blocked(self):
        self.assertIn(10, wddm_desktop_exclusions({10}, xml(), lambda pid:'Code.exe'))
        self.assertEqual(wddm_desktop_exclusions({10}, xml(), lambda pid:'python.exe'), {})
        self.assertEqual(wddm_desktop_exclusions({10}, xml(name='python.exe'), lambda pid:'Code.exe'), {})

    def test_compute_only_linux_unknown_and_unreadable_remain_blocked(self):
        for inventory,name in ((xml(kind='C'),'Code.exe'),(xml(mode='TCC'),'Code.exe'),(xml(name='unknown.exe'),'unknown.exe'),('invalid','Code.exe')):
            with self.subTest(inventory=inventory):
                self.assertEqual(wddm_desktop_exclusions({10},inventory,lambda pid:name),{})
        def denied(pid):raise PermissionError('denied')
        self.assertEqual(wddm_desktop_exclusions({10},xml(),denied),{})

    def test_stale_entry_requires_actual_os_absence(self):
        self.assertIn(10,wddm_desktop_exclusions({10},xml(name=''),lambda pid:None))
        self.assertEqual(wddm_desktop_exclusions({10},xml(name=''),lambda pid:'python.exe'),{})

    def test_vendor_utility_requires_protected_vendor_path(self):
        self.assertIn(10,wddm_desktop_exclusions({10},xml(name='GCC.exe'),lambda pid:r'C:\Program Files\GIGABYTE\Control Center\GCC.exe'))
        self.assertEqual(wddm_desktop_exclusions({10},xml(name='GCC.exe'),lambda pid:r'C:\work\GCC.exe'),{})

    def test_local_desktop_registry_requires_exact_path_and_current_hash(self):
        from unittest.mock import patch
        name=r'C:\Program Files\Example\GameViewerServer.exe'
        trusted={name.casefold():'a'*64}
        with patch('personal_llm_data.personal_runtime.desktop_file_digest',return_value='a'*64):
            self.assertIn(10,wddm_desktop_exclusions({10},xml(name='GameViewerServer.exe'),lambda pid:name,trusted))
            self.assertEqual(wddm_desktop_exclusions({10},xml(name='GameViewerServer.exe'),lambda pid:r'C:\Other\GameViewerServer.exe',trusted),{})
            self.assertEqual(wddm_desktop_exclusions({10},xml(kind='C',name='GameViewerServer.exe'),lambda pid:name,trusted),{})
        with patch('personal_llm_data.personal_runtime.desktop_file_digest',return_value='b'*64):
            self.assertEqual(wddm_desktop_exclusions({10},xml(name='GameViewerServer.exe'),lambda pid:name,trusted),{})

    def test_local_desktop_registry_cannot_allow_python(self):
        name=r'C:\python.exe'
        self.assertEqual(wddm_desktop_exclusions({10},xml(name='python.exe'),lambda pid:name,{name.casefold():'a'*64}),{})
