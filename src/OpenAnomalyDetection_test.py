from . import OpenAnomalyDetection

def test_OpenAnomalyDetection():
    assert OpenAnomalyDetection.apply("Jane") == "hello Jane"
