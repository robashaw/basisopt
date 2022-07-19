import pytest

from basisopt.parallelise import chunk

def test_chunk():
    x = [1]
    chunks = chunk(x, 3)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1
    assert len(chunks[2]) == 0
    
    x = [n for n in range(100)]
    chunks = chunk(x, 7)
    assert len(chunks) == 7
    assert len(chunks[0]) == 15
    assert len(chunks[6]) == 14
    
    chunks = chunk(x, 5)
    assert len(chunks) == 5
    assert len(chunks[2]) == 20
    
