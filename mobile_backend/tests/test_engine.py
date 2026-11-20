import os,sys,tempfile,unittest,time
from pathlib import Path
from unittest.mock import patch,Mock
from cryptography.fernet import Fernet
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from store import Store
from service import Engine,frame,config_id
from broker import floor_step,Bybit

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.key=Fernet.generate_key();self.s=Store(self.tmp.name,self.key);self.e=Engine(self.s)
        self.c=dict(symbols=['ICNTUSDT'],interval='15',mode='paper',amount=100,start=1,end=2)
        self.s.put('config',self.c);self.s.put('portfolio',dict(cash=100.,initial=100.,realized=0.,wins=0,closed=0));self.s.put('portfolio_mode','paper')
    def tearDown(self):self.e.stop();self.s.db.close();self.tmp.cleanup()
    def test_encrypted_credentials(self):
        self.s.secret('sensitive-key','sensitive-secret');self.assertEqual(self.s.secret(),['sensitive-key','sensitive-secret'])
        self.assertNotIn('sensitive',Path(self.tmp.name,'state.sqlite').read_bytes().decode(errors='ignore'))
    def test_not_approved_blocks_start(self):
        with self.assertRaisesRegex(ValueError,'لم يجتز'):self.e.start()
    def test_real_requires_server_flag_and_confirmation(self):
        self.c['mode']='real';self.s.put('config',self.c)
        with patch.dict(os.environ,{'ENABLE_REAL_TRADING':'false'}):
            with self.assertRaises(ValueError):self.e.start(True)
    def test_paper_never_posts(self):
        broker=Mock();self.e.execute('ICNTUSDT','Buy',20,1,self.c,broker,atr=.02)
        broker.order.assert_not_called();p=self.s.get('positions')['ICNTUSDT'];self.assertAlmostEqual(self.s.get('portfolio')['cash'],80)
        self.e.execute('ICNTUSDT','Sell',p['qty'],1.1,self.c,broker)
        self.assertFalse(self.s.get('positions'));self.assertGreater(self.s.get('portfolio')['realized'],0)
    def test_unique_fill_not_applied_twice(self):
        self.e.apply_fill('ICNTUSDT','Buy',20,20,self.c,.02,'unique')
        self.e.apply_fill('ICNTUSDT','Buy',20,20,self.c,.02,'unique')
        self.assertEqual(self.s.get('portfolio')['cash'],80);self.assertEqual(len(self.s.get('ledger')),1)
    def test_timeout_is_not_retried(self):
        self.c['mode']='demo';broker=Mock();broker.instruments.return_value=[dict(symbol='ICNTUSDT',lotSizeFilter=dict(quotePrecision='.01',basePrecision='.001',minOrderAmt='5'))];broker.order.side_effect=TimeoutError()
        with self.assertRaises(TimeoutError):self.e.execute('ICNTUSDT','Buy',20,1,self.c,broker,atr=.02)
        self.assertEqual(broker.order.call_count,1);self.assertIsNotNone(self.s.get('pending'))
    def test_stop_preserves_holdings(self):
        self.e.execute('ICNTUSDT','Buy',20,1,self.c,Mock(),atr=.02);self.e.stop();self.assertTrue(self.s.get('positions'));self.assertFalse(self.s.get('running'))
    def test_snapshot_restores_after_crash(self):
        self.e.apply_fill('ICNTUSDT','Buy',20,20,self.c,.02,'one');self.s.put('ledger',[]);self.s.put('portfolio',{})
        Engine(self.s);self.assertEqual(self.s.get('portfolio')['cash'],80);self.assertEqual(len(self.s.get('ledger')),1)
    def test_gap_rejected(self):
        rows=[[1700000100000+i*900000,1,2,.5,1,5] for i in range(301)];rows.pop(100)
        with self.assertRaisesRegex(ValueError,'فجوات'):frame(rows,'15')
    def test_decimal_precision(self):self.assertEqual(floor_step(1.234567,'.001'),'1.234')
    def test_signing_request_no_leverage(self):
        b=Bybit('a','b','demo')
        with patch.object(b,'request',return_value={}) as send:
            b.order('ICNTUSDT','Buy','10','id');data=send.call_args.args[2]
            self.assertEqual(data['category'],'spot');self.assertEqual(data['isLeverage'],0);self.assertEqual(data['marketUnit'],'quoteCoin')
