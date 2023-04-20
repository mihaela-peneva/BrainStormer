import unittest
from django.test import RequestFactory
from stormerapp.views import mainPage, descriptionPage, generatorPage, isFileCsv
import os
from django.conf import settings



class TestMainPageView(unittest.TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        file_path = os.path.join(settings.BASE_DIR, 'test_data', 'dummy1.csv')


    def test_mainPage(self):
        request = self.factory.get('/')
        response = mainPage(request)
        self.assertEqual(response.status_code, 200)

    def test_descriptionPage(self):
        request = self.factory.get('/description')
        response = descriptionPage(request)
        self.assertEqual(response.status_code, 200)

    def test_valid_file(self):
        file_path = os.path.join(settings.BASE_DIR, 'test_data', 'dummy1.csv')
        request = self.factory.post('/generator', {
            'customFile': open(file_path, 'rb'),
            'channels': ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5'],
        })
        response = generatorPage(request)
        self.assertEqual(response.status_code, 200)

    def test_invalid_file(self):
        file_path = os.path.join(settings.BASE_DIR, 'test_data', 'dummy2.txt')
        request = self.factory.post('/generator', {
            'customFile': open(file_path, 'rb'),
            'channels': ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5'],
        })
        response = generatorPage(request)
        self.assertEqual(response.status_code, 200)

    def test_isFileCsv(self):
        class FileObj:
            def __init__(self, name):
                self.name = name

        valid_file = FileObj('valid.csv')
        invalid_file = FileObj('invalid.txt')

        self.assertTrue(isFileCsv(valid_file))
        self.assertFalse(isFileCsv(invalid_file))


if __name__ == '__main__':
    unittest.main()
