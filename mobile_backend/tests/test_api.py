import os,unittest,tempfile
from unittest.mock import patch
from cryptography.fernet import Fernet
os.environ['ENGINE_TOKEN']='a'*40
os.environ['ENGINE_ENCRYPTION_KEY']=Fernet.generate_key().decode()
_TMP=tempfile.TemporaryDirectory();os.environ['ENGINE_DATA']=_TMP.name
from fastapi.testclient import TestClient
import app
class ApiTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app.app);self.headers={'Authorization':'Bearer '+'a'*40}
    def test_auth(self):self.assertEqual(self.client.get('/v1/status').status_code,401)
    def test_status_has_no_secrets(self):
        app.s.secret('hide-key','hide-secret');r=self.client.get('/v1/status',headers=self.headers);self.assertEqual(r.status_code,200);self.assertNotIn('hide-',r.text)
    def test_config_rejects_futures_and_invalid_dates(self):
        c=dict(symbols=['ICNTUSDT'],mode='linear',amount=100,interval='15',start=1,end=2)
        self.assertEqual(self.client.post('/v1/config',json=c,headers=self.headers).status_code,422)
    def test_config_and_stop(self):
        c=dict(symbols=['ICNTUSDT'],mode='paper',amount=100,interval='15',start=1700000000000,end=1700010000000)
        with patch('app.Bybit.instruments',return_value=[{'symbol':'ICNTUSDT'}]):self.assertEqual(self.client.post('/v1/config',json=c,headers=self.headers).status_code,200)
        self.assertEqual(self.client.post('/v1/stop',json={},headers=self.headers).status_code,200)

    def test_paper_only_host_rejects_real(self):
        c=dict(symbols=['ICNTUSDT'],mode='real',amount=100,interval='15',start=1,end=2)
        with patch.dict(os.environ,{'PAPER_ONLY':'true'}):
            self.assertEqual(self.client.post('/v1/config',json=c,headers=self.headers).status_code,422)
    def test_health_has_no_private_state(self):
        self.assertEqual(self.client.get('/healthz').json(),{'status':'ok'})
