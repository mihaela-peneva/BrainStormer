from django.test import TestCase
import io
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from unittest import TestCase
from unittest.mock import MagicMock
import sys
sys.modules['mne'] = mne # Mock mne module so that it doesn't cause issues during import
from stormerapp.views import generatorPage 

class TestGeneratorPage(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def create_csv_file(self):
        csvfile = io.StringIO()
        csvfile.write("timestamps,eeg_1,eeg_2,eeg_3,eeg_4\n")
        csvfile.write("0,1,2,3,4\n")
        csvfile.write("1,2,3,4,5\n")
        csvfile.seek(0)
        return csvfile

    def test_generator_page(self):
        # Create CSV file for testing
        csvfile = self.create_csv_file()
        uploaded_file = SimpleUploadedFile("test.csv", csvfile.getvalue().encode(), content_type="text/csv")

        # Create request
        request = self.factory.post('/generator/', {'customFile': uploaded_file})
        request.FILES['customFile'] = uploaded_file

        print(uploaded_file)
        # Call the view function
        response = generatorPage(request)

        # Check response
        self.assertEqual(response.status_code, 200)

        # Check if output file is created
        output_image_path = 'stormerapp/static/images/output.png'
        self.assertTrue(os.path.isfile(output_image_path))

        # Clean up
        os.remove(output_image_path)
