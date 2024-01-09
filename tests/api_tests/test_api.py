import os
import unittest
from unittest.mock import patch

import jwt
import requests
from dotenv import get_key

from supervisely.api.api import Api, LoginInfo


class TestLoginInfo(unittest.TestCase):
    def setUp(self):
        self.server = "https://dev.supervisely.com"
        self.invalid_server = "invalid_server"
        self.login = "unit_tests"
        self.password = "xxxxxxxxxxxxxxx"  # write your password here
        self.login_info = LoginInfo(self.server, self.login, self.password)

    def test_validate_server_url(self):
        self.assertTrue(self.login_info._validate_server_url(self.server))
        self.assertFalse(self.login_info._validate_server_url(self.invalid_server))

    def test_login_failed(self):
        with patch.object(requests, "post") as mock_post:
            mock_post.return_value.status_code = 401
            with self.assertRaises(RuntimeError):
                self.login_info.log_in()
                self.assertTrue(mock_post.called)

    def test_login_successful(self):
        with patch.object(requests, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"token": "jwt_token"}
            with patch.object(jwt, "decode") as mock_decode:
                mock_decode.return_value = {"username": self.login, "email": "test@example.com"}
                result = self.login_info.log_in()
                self.assertEqual(result.username, self.login)
                self.assertEqual(result.email, "test@example.com")
                self.assertTrue(mock_post.called)
                self.assertTrue(mock_decode.called)


class TestApi(unittest.TestCase):
    def setUp(self):
        self.server = "https://dev.supervisely.com"
        self.login = "unit_tests"  # write your login here
        self.password = "xxxxxxxxxxxxxxx"  # write your password here
        self.api_token = LoginInfo(self.server, self.login, self.password).log_in().api_token
        self.env_file = "./supervisely.env"
        with open(self.env_file, "w") as file:
            file.write(f'SERVER_ADDRESS="{self.server}"\n')
            file.write(f'API_TOKEN="{self.api_token}"\n')

    def test_from_credentials_file_exists(self):
        with patch("supervisely.api.SUPERVISELY_ENV_FILE", self.env_file):
            with self.assertRaises(RuntimeError):
                Api.from_credentials(self.server, self.login, self.password)
            api = Api.from_credentials(self.server, self.login, self.password, is_overwrite=True)
            os.path.isfile(self.env_file + ".bak")
            self.assertEqual(api.user.get_my_info().login, self.login)
            self.assertEqual(get_key(self.env_file, "API_TOKEN"), self.api_token)
            self.assertEqual(get_key(self.env_file, "SERVER_ADDRESS"), self.server)
            os.remove(self.env_file + ".bak")
            os.remove(self.env_file)


if __name__ == "__main__":
    unittest.main()
