import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.types import SemanticMapOutput
from tests.base_test import BaseTest


class TestSemMap(BaseTest):
    def test_sem_map_basic(self):
        """Test basic sem_map functionality with a single sample."""
        lm = LM(model="gpt-4o-mini")
        lotus.settings.configure(lm=lm)
        
        data = {
            "Course Name": [
                "Probability and Random Processes",
                "Computer Security",
            ]
        }
        df = pd.DataFrame(data)
        
        # Test basic map functionality
        result_df = df.sem_map("What is a similar course to {Course Name}. Be concise.")
        
        # Check that the output has the expected structure
        assert "_map" in result_df.columns
        assert len(result_df) == len(df)
        
        # Check that all outputs are non-empty strings
        for output in result_df["_map"]:
            assert isinstance(output, str)
            assert len(output) > 0
    
    def test_sem_map_multiple_samples(self):
        """Test sem_map with multiple samples."""
        lm = LM(model="gpt-4o-mini")
        lotus.settings.configure(lm=lm)
        
        data = {
            "Course Name": [
                "Probability and Random Processes",
                "Computer Security",
            ]
        }
        df = pd.DataFrame(data)
        
        # Test with multiple samples
        nsample = 3
        result_df = df.sem_map(
            "What is a similar course to {Course Name}. Be concise.",
            nsample=nsample,
            temperature=0.7
        )
        
        # Check that the output has the expected columns
        for i in range(1, nsample + 1):
            assert f"_map_{i}" in result_df.columns
        
        # Check that the "_map_all" column exists and has the right structure
        assert "_map_all" in result_df.columns
        
        # Each row in _map_all should be a list of nsample elements
        for row_samples in result_df["_map_all"]:
            assert isinstance(row_samples, list)
            assert len(row_samples) == nsample
            
            # Each sample should be a non-empty string
            for sample in row_samples:
                assert isinstance(sample, str)
                assert len(sample) > 0
    
    def test_sem_map_explanations(self):
        """Test sem_map with return_explanations=True for both single and multiple samples."""
        lm = LM(model="gpt-4o-mini")
        lotus.settings.configure(lm=lm)
        
        data = {
            "Course Name": [
                "Probability and Random Processes",
                "Computer Security",
            ]
        }
        df = pd.DataFrame(data)
        
        # Test with explanations for single sample
        result_df_single = df.sem_map(
            "What is a similar course to {Course Name}. Be concise.",
            return_explanations=True
        )
        
        # Check that the explanation column exists for single sample
        assert "explanation_map" in result_df_single.columns
        
        # Test with explanations for multiple samples
        nsample = 2
        result_df_multi = df.sem_map(
            "What is a similar course to {Course Name}. Be concise.",
            nsample=nsample,
            return_explanations=True
        )
        
        # Check that the explanation columns exist for each sample
        for i in range(1, nsample + 1):
            assert f"explanation_map_{i}" in result_df_multi.columns 