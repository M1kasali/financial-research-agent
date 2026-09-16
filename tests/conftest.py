import socket

import pytest
from agent.schemas import Chunk

from financial_agent.corpus import Corpus


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def corpus():
    return Corpus(
        [
            Chunk("a1", "a", "financial_reports", 1, "收入", "", "甲公司2025年营业收入120万元。"),
            Chunk("a2", "a", "financial_reports", 2, "收入", "", "甲公司2024年营业收入100万元。"),
            Chunk("b1", "b", "financial_contracts", 3, "终止", "", "乙合同允许提前终止，但需提前30日通知。"),
        ]
    )
