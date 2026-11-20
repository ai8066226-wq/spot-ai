import json,os,sqlite3,threading
from pathlib import Path
from cryptography.fernet import Fernet
class Store:
    def __init__(self,root,key):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.cipher=Fernet(key);self.lock=threading.RLock()
        self.db=sqlite3.connect(self.root/'state.sqlite',check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY,v TEXT)');self.db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT DEFAULT CURRENT_TIMESTAMP,message TEXT)');self.db.commit()
    def get(self,k,default=None):
        with self.lock:
            r=self.db.execute('SELECT v FROM kv WHERE k=?',(k,)).fetchone();return json.loads(r[0]) if r else default
    def put(self,k,v):
        with self.lock:self.db.execute('INSERT OR REPLACE INTO kv VALUES(?,?)',(k,json.dumps(v,allow_nan=False)));self.db.commit()
    def secret(self,key=None,secret=None,mode='real'):
        name='credentials_'+mode
        if key is not None:self.put(name,self.cipher.encrypt(json.dumps([key,secret]).encode()).decode())
        raw=self.get(name);return json.loads(self.cipher.decrypt(raw.encode())) if raw else ['','']
    def log(self,message):
        with self.lock:self.db.execute('INSERT INTO events(message) VALUES(?)',(message[:1000],));self.db.commit()
    def events(self):
        with self.lock:return [dict(id=r[0],at=r[1],message=r[2]) for r in self.db.execute('SELECT id,at,message FROM events ORDER BY id DESC LIMIT 100')]
