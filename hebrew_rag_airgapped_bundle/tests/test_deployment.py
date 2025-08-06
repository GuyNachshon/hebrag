import pytest
import requests
import time
import os

class TestDeployment:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Wait for services to be ready"""
        time.sleep(5)
    
    def test_system_health(self):
        """Test system health endpoint"""
        response = requests.get("http://localhost:8000/system-status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agno_version" in data
    
    def test_upload_endpoint(self):
        """Test document upload endpoint"""
        # This would require a test document
        pass
    
    def test_question_endpoint(self):
        """Test Hebrew question answering"""
        payload = {
            "question": "בדיקת מערכת בעברית"
        }
        response = requests.post("http://localhost:8000/ask-question", json=payload)
        assert response.status_code == 200
    
    def test_environment_variables(self):
        """Test air-gapped environment settings"""
        assert os.environ.get("AGNO_TELEMETRY") == "false"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

if __name__ == "__main__":
    pytest.main([__file__])
