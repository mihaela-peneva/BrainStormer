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
from unittest.mock import MagicMock, patch
import sys
import csv
sys.modules['mne'] = mne # Mock mne module so that it doesn't cause issues during import
from stormerapp.views import generatorPage 

class TestGeneratorPage(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def create_csv_file(self):
        # does not create the file properly
        csvfile = io.StringIO()
        csvfile.write("timestamps,eeg_1,eeg_2,eeg_3,eeg_4\n")
        csvfile.write("0,1,2,3,4\n")
        csvfile.write("1,2,3,4,5\n")

        # def read_csv(file):
        #     reader = csv.reader(file)
        #     header = next(reader)
        #     data = [row for row in reader]
        #     return header, data


        # Set the return_value of the mock_csv_reader to the actual CSV reader for the mocked file content
        mock_csv_reader.return_value = csv.reader(file_content)

        # Call the read_csv function with the mocked CSV reader
        header, data = read_csv(file_content)

        # Perform assertions
        self.assertEqual(header, ['Name', 'Age', 'Email'])
        self.assertEqual(data, [['Alice', '28', 'alice@example.com'], ['Bob', '33', 'bob@example.com']])


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
