"""Tests for retrieval components."""

import pytest
import numpy as np
from src.retrieval.document_store import DocumentStore
from src.utils.text_processing import TextProcessor


class TestDocumentStore:
    """Test cases for DocumentStore."""
    
    @pytest.fixture
    def store(self):
        """Create a document store instance."""
        return DocumentStore(vector_dimension=10)
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding."""
        return np.random.randn(10).astype(np.float32)
    
    def test_store_initialization(self, store):
        """Test store initialization."""
        assert store.vector_dimension == 10
        assert store.get_size() == 0
    
    def test_add_document(self, store, sample_embedding):
        """Test adding a document."""
        store.add_document(
            content="Test document",
            embedding=sample_embedding,
            metadata={"source": "test"}
        )
        assert store.get_size() == 1
    
    def test_invalid_embedding_dimension(self, store):
        """Test that invalid embedding dimension raises error."""
        invalid_embedding = np.random.randn(5)
        with pytest.raises(ValueError):
            store.add_document(
                content="Test",
                embedding=invalid_embedding
            )
    
    def test_clear_store(self, store, sample_embedding):
        """Test clearing the store."""
        store.add_document("Doc 1", sample_embedding)
        store.add_document("Doc 2", sample_embedding)
        assert store.get_size() == 2
        
        store.clear()
        assert store.get_size() == 0


class TestTextProcessor:
    """Test cases for TextProcessor."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  Hello   world  \n  test  "
        clean = TextProcessor.clean_text(dirty_text)
        assert clean == "Hello world test"
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "a" * 1000
        chunks = TextProcessor.chunk_text(text, chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 100
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "Hello world. This is a test! How are you?"
        sentences = TextProcessor.split_sentences(text)
        
        assert len(sentences) == 3
        assert all(s for s in sentences)  # No empty sentences
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Python is great. Python is powerful. Python is popular."
        keywords = TextProcessor.extract_keywords(text, num_keywords=5)
        
        assert len(keywords) <= 5
        assert "python" in keywords or "powerful" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
