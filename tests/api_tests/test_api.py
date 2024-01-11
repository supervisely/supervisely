import glob
import os
import sys
import unittest
from unittest.mock import patch

import jwt
import requests
from dotenv import get_key

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)
from supervisely.api.api import Api, UserSession


class TestLoginInfo(unittest.TestCase):
    def setUp(self):
        self.server = "https://app.supervisely.com"
        self.invalid_server = "invalid_server"
        self.login = "unit_tests"
        self.password = "xxxxxxxxxxxxxx"  # write your password here
        self.session = UserSession(self.server)

    def test_validate_server_url(self):
        self.assertTrue(self.session._validate_server_url(self.server))
        self.assertFalse(self.session._validate_server_url(self.invalid_server))

    def test_login_failed(self):
        with patch.object(requests, "post") as mock_post:
            mock_post.return_value.status_code = 401
            with self.assertRaises(RuntimeError):
                self.session.log_in(self.login, self.password)
                self.assertTrue(mock_post.called)

    def test_login_successful(self):
        with patch.object(requests, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"token": "jwt_token"}
            with patch.object(jwt, "decode") as mock_decode:
                mock_decode.return_value = {"username": self.login, "email": "test@example.com"}
                result = self.session.log_in(self.login, self.password)
                self.assertEqual(result.username, self.login)
                self.assertEqual(result.email, "test@example.com")
                self.assertTrue(mock_post.called)
                self.assertTrue(mock_decode.called)


class TestApi(unittest.TestCase):
    def setUp(self):
        self.server = "https://app.supervisely.com"
        self.login = "unit_tests"  # write your login here
        self.password = "xxxxxxxxxxxxxx"  # write your password here
        self.api_token = UserSession(self.server).log_in(self.login, self.password).api_token
        self.env_file = "./supervisely.env"
        with open(self.env_file, "w") as file:
            file.write(f'SERVER_ADDRESS="{self.server}"\n')
            file.write(f'API_TOKEN="{self.api_token}"\n')

    def test_from_credentials_file_exists(self):
        self.env_file = os.path.abspath(self.env_file)
        with patch("supervisely.api.SUPERVISELY_ENV_FILE", self.env_file):
            with self.assertRaises(RuntimeError):
                login = "unit_tests_1"
                password = "xxxxxxxxxxxxxx"  # write your password here
                Api.from_credentials(self.server, login, password)
            api = Api.from_credentials(self.server, self.login, self.password, is_overwrite=True)
            self.assertTrue(7 > len(glob.glob(f"{self.env_file}*")) > 1)
            self.assertEqual(api.user.get_my_info().login, self.login)
            self.assertEqual(get_key(self.env_file, "API_TOKEN"), self.api_token)
            self.assertEqual(get_key(self.env_file, "SERVER_ADDRESS"), self.server)
            for item in glob.glob(f"{self.env_file}*"):
                os.remove(item)


if __name__ == "__main__":
    unittest.main()
