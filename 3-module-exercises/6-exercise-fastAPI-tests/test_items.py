from cgi import test
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_items_correct():
    r = client.get('/items/100')
    print(r.json())
    assert r.status_code == 200
    assert r.json()['fetch'] == 'Fetched 1 of 100'

def test_items_query_correct():
    r = client.get('/items/10?count=10')
    print(r.json())
    assert r.status_code == 200
    assert r.json()['fetch'] == 'Fetched 10 of 10'

def test_items_other_query():
    r = client.get('/items/20?invlalid=20')
    print(r.json())
    assert r.status_code == 200
    assert r.json()['fetch'] == 'Fetched 1 of 20'

def test_items_incorrect_url():
    r = client.get('/items/')
    print(r.json())
    assert r.status_code == 404
    
def test_items_incorrect_type():
    r = client.get('/items/string')
    print(r.json())
    assert r.status_code == 422


def test_items_incorrect_query_type():
    r = client.get('/items/20?count=10.32')
    print(r.json())
    assert r.status_code == 422
